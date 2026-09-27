# Global cache for the preloaded model
_interface = None
_device = None

def set_interface(interface, device):
    global _interface, _device
    _interface = interface
    _device = device

def get_interface():
    return _interface, _device

def has_interface():
    return _interface is not None
