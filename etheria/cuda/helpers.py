import etheria._eth as _eth

def is_cuda_available() -> bool:
    """
    Check if CUDA is available on the system.
    """
    return _eth.is_cuda_available()

def number_cuda_devices() -> int:
    """
    Get the number of CUDA devices available.
    """
    return _eth.number_cuda_devices()

def list_cuda_devices() -> list[str]:
    """
    List all available CUDA devices.
    """
    return _eth.list_cuda_devices()

def set_cuda_device(device: int) -> None:
    """
    Set the active CUDA device.
    """
    _eth.set_cuda_device(device)

def current_cuda_device() -> int:
    """
    Get the current active CUDA device.
    """
    return _eth.current_cuda_device()

def device_details(device: int) -> str:
    """
    Get details about a specific CUDA device.
    """
    return _eth.device_details(device)