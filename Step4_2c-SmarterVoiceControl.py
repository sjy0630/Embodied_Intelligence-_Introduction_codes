import sounddevice as sd
import numpy as np
import requests
import json
import time
import os
import sys
# 引入 FunASR (SenseVoiceSmall)
from funasr import AutoModel
# 引入 sherpa_onnx (仅用于 VAD 断句)
from sherpa_onnx import VadModelConfig, SileroVadModelConfig, VoiceActivityDetector

# ================= 配置区 =================
# 1. 模型路径
# (请确保你的 model/VAD/silero_vad.onnx 文件存在)
vad_path = 'model/VAD' 

# 2. 小车 IP 地址 (请修改为你树莓派的实际 IP)
control_url = "http://10.207.27.17:5000/control" 

# 3. DeepSeek API 配置
API_KEY = "sk-82c484322c3f452fb8b54858ad1cc54f"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-chat"
# ==========================================

# HTTP Session (保持长连接，减少延迟)
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
})

# 全局状态记忆 (用于处理 "加速" 这种不带方向的指令)
last_command_state = {"command": "STOP", "steer": 0.0, "throttle": 0.0}

# --- 1. 初始化 SenseVoiceSmall 模型 ---
print("正在加载 SenseVoiceSmall 模型 (首次运行会自动下载，约500MB)...")
try:
    # device="cuda" 如果有N卡建议用 cuda，没有则用 cpu
    asr_model = AutoModel(
        model="iic/SenseVoiceSmall",
        device="cuda" if np.mod(1,1)==0 else "cpu", 
        disable_update=True,
        log_level="ERROR"
    )
    print("✅ SenseVoiceSmall 加载完成！")
except Exception as e:
    print(f"❌ ASR 模型加载失败: {e}")
    print("请确保已安装: pip install funasr modelscope torch")
    exit()

# --- 2. 初始化 VAD ---
try:
    config = VadModelConfig(
        SileroVadModelConfig(
            model=f'{vad_path}/silero_vad.onnx',
            min_silence_duration=0.5, # 0.5秒静音视为一句话结束
            threshold=0.5
        ),
        sample_rate=16000
    )
    vad = VoiceActivityDetector(config, buffer_size_in_seconds=100)
except Exception as e:
    print(f"❌ VAD 加载失败: {e}")
    print(f"请检查路径: {vad_path}/silero_vad.onnx")
    exit()

# --- 3. System Prompt (慢速安全版) ---
SYSTEM_PROMPT = """
你是一个智能小车的控制大脑。请将用户的口语指令转换为 JSON 控制信号。

### 接口定义
1. command: ["FORWARD", "BACKWARD", "STOP"]
   - FORWARD: 前进 或 转向
   - BACKWARD: 后退
   - STOP: 停止
2. steer: 浮点数 -1.0(最左) 到 1.0(最右)。0.0(直)。
   - "左转": -1.0
   - "左转一点点/微调": -0.3
   - "右转": 1.0
3. throttle: 0.0 到 1.0 (速度)。
   - "慢/一点点": 0.15 - 0.2 (非常慢)
   - "正常": 0.35 (安全速度)
   - "快/加速": 0.6 (不要太快)

### 规则
1. "停", "别动" -> STOP。
2. "不要左转" (否定) -> STOP。
3. 输出 JSON。

示例: {"command": "FORWARD", "steer": 0.0, "throttle": 0.35}
"""

def parse_local_fast(text: str):
    """
    本地快速解析 (已调整为慢速参数)
    """
    global last_command_state
    
    # 1. 紧急停止
    if any(w in text for w in ["停", "刹车", "别动", "stop"]):
        return {"command": "STOP", "steer": 0.0, "throttle": 0.0}

    # 2. 识别方向
    new_cmd = "FORWARD" 
    if "后" in text or "退" in text: new_cmd = "BACKWARD"
    
    # 3. 识别转向
    new_steer = 0.0
    if "左" in text: new_steer = -1.0
    elif "右" in text: new_steer = 1.0

    # 4. 识别速度 (默认慢速)
    new_throttle = 0.35 
    if any(w in text for w in ["慢", "缓", "小", "微"]): 
        new_throttle = 0.2
    elif any(w in text for w in ["快", "速", "冲", "急"]): 
        new_throttle = 0.6

    # 5. 处理纯速度/方向指令 (继承逻辑)
    has_direction = any(w in text for w in ["前", "后", "左", "右", "走", "退"])
    if not has_direction:
        # 如果没说方向，继承上一次的前后状态
        if last_command_state['command'] == "BACKWARD": new_cmd = "BACKWARD"
        else: new_cmd = "FORWARD"
        new_steer = 0.0 # 纯加速时默认回正，防止画圈
    
    return {"command": new_cmd, "steer": new_steer, "throttle": new_throttle}

def get_command_from_llm(text: str) -> dict:
    """调用 DeepSeek API"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        "stream": False,
        "temperature": 0.1, 
        "response_format": {"type": "json_object"} 
    }
    
    print(f"🤖 DeepSeek 思考: '{text}' ...")
    try:
        response = session.post(API_URL, json=payload, timeout=3)
        if response.status_code != 200:
            print(f"⚠️ API 错误: {response.text}")
            return None
        
        # [修复] 之前这里被截断了，现在补全了逻辑
        r = response.json()['choices'][0]['message']['content'].strip()
        if "```" in r: 
            r = r.replace("```json", "").replace("```", "")
        
        return json.loads(r)
    except Exception as e:
        print(f"❌ LLM 失败: {e}")
        return None

def send_to_car(json_cmd):
    """发送指令给树莓派"""
    global last_command_state
    if not json_cmd: return
    
    last_command_state = json_cmd
    
    # 打印进度条
    spd = json_cmd.get('throttle', 0)
    spd_bar = "█" * int(spd * 20)
    print(f"🚀 发送: {json_cmd['command']} | 转向:{json_cmd.get('steer')} | 速度:{spd:.2f} {spd_bar}")
    
    try:
        requests.post(control_url, json=json_cmd, timeout=1)
    except Exception as e:
        print(f"通信错误: {e}")

# --- 主循环 ---
print('\n🎙️ 高精度语音控制 (SenseVoice + DeepSeek) 已启动... (Ctrl+C 退出)')
sample_rate = 16000
samples_per_read = int(0.1 * sample_rate) 

try:
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)
            samples = samples.reshape(-1)
            
            # VAD 断句
            vad.accept_waveform(samples)
            
            if not vad.empty():
                # [Fix] 确保转换为 numpy array
                audio_segment = np.array(vad.front.samples)
                vad.pop()
                
                # SenseVoice 识别
                if len(audio_segment) > 0:
                    try:
                        res = asr_model.generate(
                            input=[audio_segment], # [Fix] 加上 [] 包装成 list，防止 funasr 把 1D 数组当成 batch 遍历导致 float 报错
                            cache={}, 
                            language="zh", 
                            use_itn=True,
                            batch_size_s=60
                        )
                        
                        text = ""
                        if isinstance(res, list) and len(res) > 0:
                            text = res[0].get("text", "")
                        
                        import re
                        text = re.sub(r'<\|.*?\|>', '', text).strip()

                        if len(text) > 0:
                            print(f"\n👂 听到: {text}")
                            
                            # 1. 尝试本地解析 (为了快)
                            cmd = parse_local_fast(text)
                            
                            # 2. 如果是复杂句 (否定/长句)，交给 LLM
                            if "不" in text or "别" in text or len(text) > 5:
                                llm_cmd = get_command_from_llm(text)
                                if llm_cmd: cmd = llm_cmd
                            
                            if cmd: send_to_car(cmd)
                            
                    except Exception as e:
                        print(f"识别出错: {e}")

except KeyboardInterrupt:
    print('\n程序被用户中断')

finally:
    # 安全退出：强制停车
    print("\n🛑 正在强制停止小车...")
    for _ in range(3):
        send_to_car({"command": "STOP", "steer": 0.0, "throttle": 0.0})
        time.sleep(0.1)
    print("程序已安全退出。")