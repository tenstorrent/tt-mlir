// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>
#include <iostream>
#include <cstdlib>

// Vibecoded test script

int main(int argc, char** argv) {
    const char* so_path = nullptr;
    
    if (argc > 1) {
        so_path = argv[1];
    } else {
        so_path = "../../templates/python/local/example_module.cpython-311-x86_64-linux-gnu.so";
    }
    
    std::cout << "Loading .so file: " << so_path << std::endl;
    
    // Open the shared library with RTLD_LAZY | RTLD_GLOBAL
    // RTLD_GLOBAL is important for Python modules that may need to see each other
    void* handle = dlopen(so_path, RTLD_LAZY | RTLD_GLOBAL);
    
    if (!handle) {
        std::cerr << "Error loading .so: " << dlerror() << std::endl;
        return 1;
    }
    
    std::cout << "Successfully loaded .so" << std::endl;
    
    // Clear any existing error
    dlerror();
    
    // Look up the entrypoint function
    typedef int (*entrypoint_fn)();
    entrypoint_fn entrypoint = (entrypoint_fn)dlsym(handle, "entrypoint");
    
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Error finding entrypoint symbol: " << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }
    
    std::cout << "Found entrypoint function, calling it..." << std::endl;
    
    // Call the entrypoint
    int result = entrypoint();
    
    std::cout << "Entrypoint returned: " << result << std::endl;
    
    // Close the library
    dlclose(handle);
    
    return result;
}

