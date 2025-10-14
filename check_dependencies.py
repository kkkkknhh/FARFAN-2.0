import sys
import json

def check_dependencies():
    results = {}
    
    # Check Python version
    results['python_version'] = sys.version
    
    # Check critical imports
    critical_modules = [
        'torch', 'numpy', 'transformers', 
        'sentence_transformers', 'sklearn'
    ]
    
    for module in critical_modules:
        try:
            mod = __import__(module)
            results[module] = {
                'installed': True,
                'version': getattr(mod, '__version__', 'unknown')
            }
        except ImportError:
            results[module] = {'installed': False}
    
    # Check GPU availability
    try:
        import torch
        results['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results['cuda_device_count'] = torch.cuda.device_count()
            results['cuda_device_name'] = torch.cuda.get_device_name(0)
    except:
        results['cuda_available'] = False
    
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    check_dependencies()
