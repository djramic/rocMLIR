import os
import ctypes

# Setup HSA simulator environment BEFORE importing hip module
# This is critical because hip module initializes HIP runtime on import
def _setup_hsa_simulator():
    """Automatically configure HSA simulator environment if available"""
    # Try multiple possible simulator locations
    possible_sim_libs = [
        "/workspaces/jitcu_docker/_builds/Debug/lib/libhsakmtmodel.so",
        os.path.expanduser("~/mi450_simulator/libhsakmtmodel.so"),
    ]
    possible_topologies = [
        "/workspaces/jitcu_docker/topology",
        os.path.expanduser("~/mi450_simulator/topology/mi450"),
    ]
    
    simulator_lib = None
    simulator_topology = None
    
    for lib_path in possible_sim_libs:
        if os.path.exists(lib_path):
            simulator_lib = lib_path
            break
    
    for topo_path in possible_topologies:
        if os.path.exists(topo_path):
            simulator_topology = topo_path
            break
    
    if simulator_lib is None:
        simulator_lib = os.path.expanduser("~/mi450_simulator/libhsakmtmodel.so")
        simulator_topology = os.path.expanduser("~/mi450_simulator/topology/mi450")
    
    # Only setup if simulator exists and HSA variables are not already set
    if os.path.exists(simulator_lib) and not os.environ.get('HSA_MODEL_LIB'):
        os.environ['HSA_MODEL_LIB'] = simulator_lib
        os.environ['HSA_MODEL_TOPOLOGY'] = simulator_topology
        os.environ['HSA_ENABLE_SDMA'] = '0'
        os.environ['HSA_ENABLE_INTERRUPT'] = '0'
        
        # Update LD_LIBRARY_PATH to include simulator directory
        sim_dir = os.path.dirname(simulator_lib)
        rocm_lib = '/opt/rocm/lib'
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if sim_dir not in ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{sim_dir}:{rocm_lib}:{ld_path}"
        
        # Set LD_PRELOAD - also preload the library via ctypes before hip import
        os.environ['LD_PRELOAD'] = simulator_lib
        
        # Preload the simulator library so it's available when hip imports
        try:
            ctypes.CDLL(simulator_lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # Library might already be loaded

# Initialize simulator environment BEFORE importing hip
_setup_hsa_simulator()

# Preload ROCm HIP library before importing hip module
# This is needed because LD_LIBRARY_PATH changes don't affect already-running process
def _preload_hip_library():
    """Preload libamdhip64.so before hip module import"""
    hip_lib_paths = [
        '/opt/rocm/lib/libamdhip64.so',
        '/opt/rocm/lib/libamdhip64.so.6',
    ]
    for path in hip_lib_paths:
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                pass

_preload_hip_library()

# Now import hip after environment is configured and libraries preloaded
from hip import hip


# Helper function to decode arch to its features
# Keep this in sync with mlir/lib/Dialect/Rock/Generator/AmdArchDb.cpp:mlir::rock::lookupArchInfo
def get_arch_features(arch: str):
    chip_name = arch.split(':')[0]
    if len(chip_name) < 5:
        return

    arch_features = None
    support_mfma = False
    support_wmma = False
    support_accel_fp8 = False
    major = chip_name[:-2]
    minor = chip_name[-2:]
    if major == 'gfx9':
        if minor in ['08', '0a']:
            arch_features = 'mfma|dot|atomic_add|atomic_add_f16'
        elif minor == '42':
            arch_features = 'mfma|dot|atomic_add|atomic_add_f16|direct_to_lds_32b'
            support_accel_fp8 = True
        elif minor == '50':
            arch_features = 'mfma|dot|atomic_add|atomic_add_f16|atomic_add_bf16|direct_to_lds_32b|direct_to_lds_128b|lds_transpose_load'
            support_accel_fp8 = True
        elif minor == '06':
            arch_features = 'dot'
        else:
            arch_features = 'none'
    elif major == 'gfx10':
        if minor in ['11', '13']:
            arch_features = 'atomic_fmax_f32'
        elif minor in ['10', '12'] or minor[0] == '3':
            arch_features = 'dot|atomic_fmax_f32'
        else:
            arch_features = 'atomic_fmax_f32'
    elif major == 'gfx11':
        arch_features = 'dot|atomic_add|atomic_fmax_f32|wmma'
    elif major == 'gfx12':
        arch_features = 'dot|atomic_add|atomic_add_f16|atomic_add_bf16|atomic_fmax_f32|wmma'
        support_accel_fp8 = True
    if arch_features and 'mfma' in arch_features:
        support_mfma = True
        pass
    elif arch_features and 'wmma' in arch_features:
        support_wmma = True
        pass
    return arch_features, support_mfma, support_wmma, support_accel_fp8


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def get_agents():
    agents = set()
    device_count = hip_check(hip.hipGetDeviceCount())
    for device in range(device_count):
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, device))
        agent = props.gcnArchName.decode('utf-8')
        agents.add(agent)

    return agents


def is_xdlops_present() -> bool:
    """This function checks whether a GPU with xdlops support is present"""
    return any([agent.startswith("gfx9") for agent in get_agents()])
