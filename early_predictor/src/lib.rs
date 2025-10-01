use std::ffi::{c_void, CString};
use std::sync::Mutex;
use std::ffi::{c_char, CStr};
use lazy_static::lazy_static;
use std::collections::HashMap;

pub mod feature_extraction;

lazy_static! {
    static ref OUTPUT_BUFFER: Mutex<String> = Mutex::new(String::new());
}

#[no_mangle]
pub unsafe extern "C" fn capture_abc_output(output: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(output) };
    let output_str = c_str.to_str().unwrap_or("");
    OUTPUT_BUFFER.lock().unwrap().push_str(output_str);
}

extern "C" {
    // fn Abc_Start();
    fn Abc_Stop();
    fn Abc_FrameGetGlobalFrame() -> *mut c_void;
    fn Cmd_CommandExecute(pAbc: *mut c_void, sCommand: *const c_char) -> i32;
}

pub struct Abc {
    ptr: *mut c_void,
}

impl Drop for Abc {
    fn drop(&mut self) {
        unsafe { Abc_Stop() };
    }
}

impl Abc {
    pub fn new() -> Self {
        // unsafe { Abc_Start() };
        let ptr = unsafe { Abc_FrameGetGlobalFrame() };
        assert!(!ptr.is_null(), "ABC initialization failed");
        Self { ptr }
    }

    pub fn execute_command(&mut self, command: &str) {
        let c = CString::new(command).expect("Command string contains null bytes");
        let res = unsafe { Cmd_CommandExecute(self.ptr, c.as_ptr()) };
        // assert_eq!(res, 0, "ABC command '{}' failed with code {}", command, res);
    }

    pub fn execute_command_with_output(&mut self, command: &str) -> String {
        OUTPUT_BUFFER.lock().unwrap().clear();
        self.execute_command(command);
        OUTPUT_BUFFER.lock().unwrap().clone()
    }
}