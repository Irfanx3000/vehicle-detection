# **🚗 Vehicle Speed & License Plate Detection**
This project detects vehicles in video streams, estimates their speed, and recognizes license plates.
If a vehicle exceeds the configured speed limit, it is logged along with its plate number and images.
Emergency vehicles can also be exempted from overspeeding detection when trained datasets are used.

```bash
pip install ultralytics
pip install easyocr
python detect.py
```
👀 Making MARK1 & MARK2 Visible

If you want to see the detection lines, add these lines in the process_video() function before the vehicle
# Debug lines for MARK1 and MARK2
```python
cv2.line(display, (0, MARK1), (WIDTH, MARK1), (0, 0, 255), 2)
cv2.line(display, (0, MARK2), (WIDTH, MARK2), (255, 0, 0), 2)
```
