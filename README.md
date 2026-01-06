# Wandless - Shortcuts at your fingertips

Wandless is a program that allows the user to translate hand gestures into useful shortcuts.

### Installation
1. **Clone the repository:**
    ```
    git clone https://github.com/YiMingPan-Xv/wandless.git
    cd wandless
    ```
2. **Setup a Virtual Environment (Recommended):**
    ```
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3. **Install dependencies:** Wandless relies on `OpenCV` for camera input and `Mediapipe` for high-fidelity hand tracking.
    ```
    pip install -r requirements.txt
    ```

### Usage

To launch the application, run the main entry point:
```
python main.py
```

A window will appear showing your camera feed. In particular, you'll find:
- A red box delimiting where the hand gesture can be detected. Outside that box, no hand movement can cause a shortcut.
- A skeleton like drawing on the detected hand. Orange nodes represent landmarks on your fingers.
- Recommendation: To test the detection, you might want to turn "always on top" option.

Triggering a shortcut is extremely easy:
- Extend your index finger, as if you're pointing upwards. Make sure the hand fits inside the frame of the camera. If it is valid, your index node will turn green.
- With a confident swipe, bring your index to one of the orthogonal directions (up, down, left, or right)
- The associated shortcut is then processed.

As of right now, only the python scripts are provided.

### Configuration

The main attraction of the program lies in the shortcuts. In `shortcuts.py`, you'll find four functions, one per each orthogonal swipe direction.

In the associated block, you can modify the shortcuts, or even add more lines for more complex tasks. By default, up and down gestures do nothing, while left and right gestures allow you to go back or forward one page in your browser (Alt + left / Alt + right).

Additionally, you can control a few settings in `config.py` for more control over the gesture detection.

