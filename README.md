# Proctoring Setup

## Steps to Run the Program

1. Create and activate a virtual environment:
   - **Linux/Mac**:
     ```bash
     python3 -m venv venv && source venv/bin/activate
     ```
   - **Windows**:
     ```bash
     python -m venv venv
     source venv/Scripts/activate
     ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

   pip install opencv-python torch torchvision matplotlib sounddevice mediapipe
   ```

3. Navigate to the source directory:
   ```bash
   cd src
   ```

4. Run the program:
   ```bash
   python run.py
   ```

To deactivate the virtual environment:
```bash
deactivate
```
