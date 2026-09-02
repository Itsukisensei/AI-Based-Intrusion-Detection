# 🚀 How to Run & Operate the Explainable AI Cloud UBA Platform

Follow either **Method 1 (1-Click Shortcut)** or **Method 2 (Command Prompt / CMD)** below.

---

## 🖱️ Method 1: The 1-Click Way (Easiest)

Inside the project folder:
1. **To Start**: Double-click [`start_dashboard.bat`](file:///c:/Users/lenovo/.gemini/antigravity-ide/scratch/Explainable-AI-Cloud-UBA/start_dashboard.bat).
   - This automatically starts the real-time telemetry streaming daemon.
   - Starts the Streamlit Cloud SOC dashboard.
   - Automatically opens the dashboard in **Opera GX** at `http://localhost:8501`.
2. **To Stop**: Double-click [`stop_dashboard.bat`](file:///c:/Users/lenovo/.gemini/antigravity-ide/scratch/Explainable-AI-Cloud-UBA/stop_dashboard.bat).
   - This cleanly terminates all background streaming and web server processes.

---

## 💻 Method 2: From Command Prompt (CMD) or PowerShell

If you or another user want to type commands manually in **CMD** or **PowerShell**:

### Step 1: Open Command Prompt (CMD)
Press `Win + R`, type `cmd`, and press **Enter**.

### Step 2: Navigate to Project Directory
```cmd
cd /d "C:\Users\lenovo\.gemini\antigravity-ide\scratch\Explainable-AI-Cloud-UBA"
```

### Step 3: Start the Real-Time Event Streamer (Background Daemon)
```cmd
start python detection\realtime_stream.py
```
*(This starts the live streaming engine in the background, writing live telemetry events every 2 seconds.)*

### Step 4: Start the Cloud SOC Dashboard
```cmd
streamlit run dashboard\app.py --server.headless=true
```

### Step 5: Open in Opera GX
In Opera GX (or from CMD):
```cmd
start "" "C:\Users\lenovo\AppData\Local\Programs\Opera GX\opera.exe" "http://localhost:8501"
```

---

## 🛑 How to Stop Everything from CMD
Whenever you want to stop the server and daemon:
```cmd
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*realtime_stream.py*' -or $_.CommandLine -like '*dashboard*app.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
```
