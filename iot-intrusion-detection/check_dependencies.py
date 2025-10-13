#!/usr/bin/env python3
"""
Check if all required dependencies are installed.
"""

def main():
    print("Checking Dependencies")
    print("=" * 30)
    
    required_packages = [
        'tensorflow', 'keras', 'numpy', 'pandas', 
        'sklearn', 'matplotlib', 'seaborn', 'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
                print(f"[OK] {package}: {sklearn.__version__}")
            elif package == 'yaml':
                import yaml
                print(f"[OK] {package}: {yaml.__version__}")
            elif package == 'tensorflow':
                import tensorflow as tf
                print(f"[OK] {package}: {tf.__version__}")
                # Check GPU availability
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    print(f"  GPU available: {len(gpus)} device(s)")
                else:
                    print(f"  GPU: Not available (CPU only)")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"[OK] {package}: {version}")
        except ImportError:
            print(f"[MISSING] {package}: Not installed")
            missing_packages.append(package)
    
    print("\n" + "=" * 30)
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("All dependencies are available!")
        print("Ready to run the pipeline!")
        return True

if __name__ == "__main__":
    main()
