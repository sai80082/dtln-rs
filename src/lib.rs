// Primary export functions for the NEON module.
use dtln_processor::DtlnDeferredProcessor;
use dtln_processor::{DtlnImmediateProcessor, DtlnProcessEngine};

use std::io::Result;
use std::ptr;
use std::slice;
use std::sync::{Arc, Mutex};
pub mod constants;
pub mod dtln_engine;
pub mod dtln_processor;
pub mod dtln_utilities;
pub mod tflite;

use neon::prelude::*;

use neon::types::buffer::TypedArray;

fn dtln_create_napi(mut cx: FunctionContext) -> JsResult<JsBox<Arc<Mutex<DtlnDeferredProcessor>>>> {
    let dtln_processor = DtlnDeferredProcessor::new();
    let Ok(dtln_processor) = dtln_processor else {
        return cx.throw_error("Failed to create DtlnDeferredProcessor");
    };

    Ok(cx.boxed(Arc::new(Mutex::new(dtln_processor))))
}

fn dtln_stop_napi(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let dtln_processor = cx.argument::<JsBox<Arc<Mutex<DtlnDeferredProcessor>>>>(0)?;
    dtln_processor.lock().unwrap().stop();
    Ok(cx.undefined())
}

#[no_mangle]
pub extern "C" fn dtln_rs_processor_create() -> *mut DtlnImmediateProcessor {
    match DtlnImmediateProcessor::new() {
        Ok(processor) => Box::into_raw(Box::new(processor)),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn dtln_rs_processor_destroy(handle: *mut DtlnImmediateProcessor) {
    if handle.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub extern "C" fn dtln_rs_denoise(
    handle: *mut DtlnImmediateProcessor,
    input_ptr: *const f32,
    len: usize,
    output_ptr: *mut f32,
) -> bool {
    if handle.is_null() || input_ptr.is_null() || output_ptr.is_null() || len == 0 {
        return false;
    }

    let processor = unsafe { &mut *handle };
    let input = unsafe { slice::from_raw_parts(input_ptr, len) };
    let output = unsafe { slice::from_raw_parts_mut(output_ptr, len) };

    match processor.denoise(input) {
        Ok(result) => {
            if result.samples.len() > output.len() {
                return false;
            }
            output[..result.samples.len()].copy_from_slice(&result.samples);
            true
        }
        Err(_) => false,
    }
}

/**
* Denoise the samples.
*
* @param {Float32Array} samples - The samples to denoise.
* @param {Float32Array} output - The denoised samples.

* @returns {boolean} - True if the processing thread is backed up.
*/
fn dtln_denoise_napi(mut cx: FunctionContext) -> JsResult<JsBoolean> {
    if cx.len() != 3 {
        return cx.throw_error("Invalid number of arguments, expected <engine: JsBox, samples: Float32Array, output: Float32Array>");
    }

    let processor_starved;

    let result: Result<()> = {
        let dtln_processor = cx.argument::<JsBox<Arc<Mutex<DtlnDeferredProcessor>>>>(0)?;

        let samples = cx.argument::<JsTypedArray<f32>>(1).unwrap();
        let mut output = cx.argument::<JsTypedArray<f32>>(2).unwrap();

        let lock = cx.lock();
        let samples_slice = samples.try_borrow(&lock).unwrap();
        let mut output_slice = output.try_borrow_mut(&lock).unwrap();

        // RefMut has to be passed up the entire chain, and I'd rather not let
        // it leak further into the dtln_denoise abstraction. Generically
        // operating on an &mut [f32] is better, so copying here is our best option.
        let denoise_result = dtln_processor
            .lock()
            .unwrap()
            .denoise(&samples_slice)
            .map_err(|e| panic!("Error in dtln_denoise: {}", e))
            .unwrap();

        processor_starved = denoise_result.processor_starved;

        output_slice[..denoise_result.samples.len()].copy_from_slice(&denoise_result.samples);
        Ok(())
    };

    if result.is_ok() {
        Ok(cx.boolean(processor_starved))
    } else {
        cx.throw_error("Error in dtln_denoise")
    }
}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("dtln_denoise", dtln_denoise_napi)?;
    cx.export_function("dtln_create", dtln_create_napi)?;
    cx.export_function("dtln_stop", dtln_stop_napi)?;

    Ok(())
}
