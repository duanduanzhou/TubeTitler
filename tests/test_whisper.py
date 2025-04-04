#!/usr/bin/env python3
import sys

print(f"Python version: {sys.version}")

try:
    import whisper
    print("Successfully imported whisper module")
    print(f"Whisper path: {whisper.__file__}")
except ImportError as e:
    print(f"Failed to import whisper: {e}")
    
    try:
        import openai.whisper as whisper
        print("Successfully imported openai.whisper module")
        print(f"Whisper path: {whisper.__file__}")
    except ImportError as e:
        print(f"Failed to import openai.whisper: {e}")

# print("\nChecking all available modules:")
import pkgutil
for module in pkgutil.iter_modules():
    if "whisper" in module.name:
        print(f"Found module: {module.name}")
