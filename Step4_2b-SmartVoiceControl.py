import sherpa_onnx
from sherpa_onnx import OfflineRecognizer
from typing import Union
import librosa
import sounddevice as sd
import numpy as np
import requests
# [新] 引入语言模型相关库
from sentence_transformers import SentenceTransformer
import joblib

# ！！！--- 请确认你的模型路径是否正确 ---！！！
asr_path = 'model/ASR/sherpa-onnx-paraformer-zh-small-2024-03-09'
vad_path = 'model/VAD'

class ASR:
    def __init__(self):
        self._recognizer = OfflineRecognizer()
        raise NotImplementedError

    def transcribe(self, audio: Union[str, np.ndarray], sample_rate=16000) -> str:
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=sample_rate)
        s = self._recognizer.create_stream()
        s.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(s)
        return s.result.text


class Whisper(ASR):
    def __init__(self, encoder_path: str, decoder_path: str, tokens_path: str, num_threads: int = 8, provider: str = 'cpu'):
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder_path,
            decoder=decoder_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=provider,
        )


class Paraformer(ASR):
    def __init__(self, model_path: str, tokens_path: str, num_threads: int = 8, provider: str = 'cpu'):
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=provider,
        )

print('正在加载ASR模型...')
try:
    asr = Paraformer(
        model_path=f'{asr_path}/model.int8.onnx',
        tokens_path=f'{asr_path}/tokens.txt',
        # provider='cuda', # 如果有显卡可以取消注释
    )
    print('ASR模型加载完成')
except Exception as e:
    print(f"ASR模型加载失败: {e}")
    exit()

print('正在加载语言模型...')
try:
    lm = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print('语言模型加载完成')
    # 请确认这个 pkl 文件路径正确
    mlp = joblib.load('model/MLP/command_classifier.pkl')
    print('指令分类器加载完成')
except Exception as e:
    print(f"语言模型或分类器加载失败: {e}")
    print("请确保已安装 sentence-transformers 和 joblib，并且模型路径正确")
    exit()

sample_rate = 16000

from sherpa_onnx import VadModelConfig, SileroVadModelConfig, VoiceActivityDetector
config = VadModelConfig(
    SileroVadModelConfig(
        model=f'{vad_path}/silero_vad.onnx',
        min_silence_duration=0.25,
    ),
    sample_rate=sample_rate
)
window_size = config.silero_vad.window_size
vad = VoiceActivityDetector(config, buffer_size_in_seconds=100)
samples_per_read = int(0.1 * sample_rate)

# ！！！--- 请确认这里的 IP 地址 ---！！！
control_url = "http://10.207.27.17:5000/control" 

def send_command(command_type):
    print(f"准备发送指令动作: {command_type}")
    
    # 默认是停止
    payload = {'command': "STOP"}
    
    try:
        if command_type == '前进':
            # throttle=1.0 表示全速，steer=0.0 表示直行
            payload = {'command': "FORWARD", 'throttle': 1.0, 'steer': 0.0}
            
        elif command_type == '后退':
            # 后退逻辑保持不变
            payload = {'command': "BACKWARD", 'throttle': -1.0, 'steer': 0.0}
            
        elif command_type == '左转':
            # --- 关键修改 ---
            # 原来的 'LEFT' 改为 'FORWARD' 加上 steer=-1.0
            # 这样才能触发树莓派的“原地掉头”逻辑
            payload = {'command': "FORWARD", 'throttle': 0.5, 'steer': -1.0}
            
        elif command_type == '右转':
            # --- 关键修改 ---
            # 原来的 'RIGHT' 改为 'FORWARD' 加上 steer=1.0
            payload = {'command': "FORWARD", 'throttle': 0.5, 'steer': 1.0}
            
        elif command_type == '停止':
            payload = {'command': "STOP", 'throttle': 0.0, 'steer': 0.0}
        
        else:
            print("未知指令类型，跳过发送")
            return

        # 发送请求
        # print(f"正在发送: {payload} -> {control_url}")
        response = requests.post(control_url, json=payload, timeout=2)
        
        if response.status_code == 200:
            print(f"✅ 小车回复: {response.text}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            
    except Exception as e:
        print('❌ 通信异常:', e)

print('\n正在识别语音指令... (按 Ctrl+C 退出)')
idx = 1
buffer = []
try:
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)

            buffer = np.concatenate([buffer, samples])
            while len(buffer) > window_size:
                vad.accept_waveform(buffer[:window_size])
                buffer = buffer[window_size:]

            while not vad.empty():
                text = asr.transcribe(vad.front.samples, sample_rate=sample_rate)

                vad.pop()
                if len(text):
                    print()
                    print(f'第{idx}句：{text}')
                    
                    # --- 1. 关键词强制拦截 ---
                    # 提高安全性：无论模型预测什么，只要听到“停”，立刻停车
                    if "停" in text or "刹车" in text or "别动" in text:
                        print('⚡ 触发关键词拦截：强制停止')
                        send_command('停止')
                        idx += 1
                        continue # 跳过后面的AI预测，直接进行下一轮
                    
                    # --- 2. AI 模型预测 ---
                    # 使用 SentenceTransformer 生成向量，再用 MLP 分类
                    try:
                        embedding = lm.encode(text).reshape(1, -1)
                        command = mlp.predict(embedding)[0]
                        
                        if command == '无操作':
                            print('未识别到有效指令 (忽略)')
                        else:
                            print('🤖 识别到 AI 指令：', command)
                            send_command(command)
                    except Exception as e:
                        print(f"模型预测出错: {e}")

                    idx += 1
except KeyboardInterrupt:
    sd.stop()
    print('\n识别结束')