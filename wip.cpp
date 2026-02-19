// Register debug callback function - provides safe single-instance nanobind access
m.def("register_debug_callback",
    []() {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
      nb::gil_scoped_acquire gil;

      std::cout << "[register_debug_callback] Importing debug_callback module..." << std::endl;

      try {
        // Import the debug_callback module
        nb::module_ debug_module = nb::module_::import_("debug_callback");

        // Get the callback functions
        nb::object pre_op_func = debug_module.attr("pre_op_callback");
        nb::object post_op_func = debug_module.attr("post_op_callback");

        // Validate callbacks are callable or None
        if (!pre_op_func.is_none() && !PyCallable_Check(pre_op_func.ptr())) {
          throw nb::type_error("pre_op_callback must be callable or None");
        }
        if (!post_op_func.is_none() && !PyCallable_Check(post_op_func.ptr())) {
          throw nb::type_error("post_op_callback must be callable or None");
        }

        std::cout << "[register_debug_callback] Callbacks loaded, registering..." << std::endl;

        // Wrap callbacks in shared_ptr with custom deleter that acquires GIL
        // This ensures Python refcounts are safely decremented when callbacks are destroyed
        // CRITICAL: Direct capture by value causes segfaults when lambda is destroyed from C++ context
        auto pre_op_shared = std::shared_ptr<nb::object>(
            new nb::object(pre_op_func),
            [](nb::object* p) {
              nb::gil_scoped_acquire gil;
              delete p;
            }
        );

        auto post_op_shared = std::shared_ptr<nb::object>(
            new nb::object(post_op_func),
            [](nb::object* p) {
              nb::gil_scoped_acquire gil;
              delete p;
            }
        );

        // Create lambda wrappers that acquire GIL and handle errors
        // Capture shared_ptr by value to keep callbacks alive
        auto pre_callback = [pre_op_shared](tt::runtime::Binary binary,
                                            tt::runtime::CallbackContext programContext,
                                            tt::runtime::OpContext opContext) {
          if (pre_op_shared->is_none()) return;

          nb::gil_scoped_acquire gil;
          try {
            (*pre_op_shared)(binary, programContext, opContext);
          } catch (nb::python_error &e) {
            std::cerr << "[register_debug_callback] Pre-callback Python error: " << e.what() << std::endl;
            e.restore();
            PyErr_Print();
          } catch (const std::exception &e) {
            std::cerr << "[register_debug_callback] Pre-callback C++ error: " << e.what() << std::endl;
          }
        };

        auto post_callback = [post_op_shared](tt::runtime::Binary binary,
                                              tt::runtime::CallbackContext programContext,
                                              tt::runtime::OpContext opContext) {
          if (post_op_shared->is_none()) return;

          nb::gil_scoped_acquire gil;
          try {
            (*post_op_shared)(binary, programContext, opContext);
          } catch (nb::python_error &e) {
            std::cerr << "[register_debug_callback] Post-callback Python error: " << e.what() << std::endl;
            e.restore();
            PyErr_Print();
          } catch (const std::exception &e) {
            std::cerr << "[register_debug_callback] Post-callback C++ error: " << e.what() << std::endl;
          }
        };

        // Register with Hooks singleton
        tt::runtime::debug::Hooks::get(pre_callback, post_callback);

        std::cout << "[register_debug_callback] ✓ Callbacks registered successfully!" << std::endl;

      } catch (nb::python_error &e) {
        std::cerr << "[register_debug_callback] Failed to import debug_callback module: " << e.what() << std::endl;
        e.restore();
        PyErr_Print();
        throw;
      } catch (const std::exception &e) {
        std::cerr << "[register_debug_callback] Failed to register callbacks: " << e.what() << std::endl;
        throw;
      }
#else
      std::cout << "[register_debug_callback] Debug mode not enabled (TT_RUNTIME_DEBUG not set)" << std::endl;
#endif
    },
    R"(
  Register debug callbacks for operation tracing.

  This function imports the debug_callback module and registers the pre_op_callback
  and post_op_callback functions defined there. Callbacks are executed with proper
  GIL management and error handling.

  The debug_callback module should define:
  - pre_op_callback(binary, program_ctx, op_ctx): Called before each operation
  - post_op_callback(binary, program_ctx, op_ctx): Called after each operation

  Notes
  -----
  - Callbacks are executed from the C++ runtime thread with GIL properly acquired
  - Python exceptions in callbacks are caught and printed without crashing execution
  - Only one set of callbacks can be active at a time (last registration wins)
  - Use `unregister_hooks()` to remove callbacks

  Examples
  --------
  >>> # In debug_callback.py:
  >>> def pre_op_callback(binary, program_ctx, op_ctx):
  ...     print(f"Op: {ttrt.runtime.get_op_debug_str(op_ctx)}")
  >>>
  >>> def post_op_callback(binary, program_ctx, op_ctx):
  ...     pass
  >>>
  >>> # Then in your code:
  >>> import ttrt.runtime
  >>> ttrt.runtime.register_debug_callback()
  )");
