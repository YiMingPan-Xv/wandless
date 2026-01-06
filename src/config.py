# The distance relative to the frame required to trigger a horizontal swipe detection
X_SWIPE_THRESHOLD = 0.20

# The distance relative to the frame required to trigger a vertical swipe detection
Y_SWIPE_THRESHOLD = 0.10

# The cooldown time in seconds (s) before another swipe can be detected
COOLDOWN_TIME = 1.5

# The minimum distance between the index and other fingers relative to the frame required for the initial gesture
FINGER_PROXIMITY_THRESHOLD = 0.1

# The maximum time in seconds (s) between the detection of the initial gesture and the non-detection of it
# If MAX_TIME_GAP seconds have passed after not detecting the initial gesture, the motion history gets wiped
MAX_TIME_GAP = 0.5

# The motion history's frame capacity
# Higher the value, the slower the swipe must be
HISTORY_LENGTH = 4

# The margin where the detection will not trigger at all
# Higher the value, the more centered must be the hand relative to the frame before the detection is active
START_MARGIN = 0.15
