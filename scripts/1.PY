import os

file_path = r"C:\Users\mouly\OneDrive\Desktop\h\dataset\Indian Sign Language Gesture Landmarks.csv"
if os.path.exists(file_path):
    print("✅ File found!")
else:
    print("❌ File not found! Check the path.")
