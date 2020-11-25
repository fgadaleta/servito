use onnxruntime::{environment::Environment,
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel,
    LoggingLevel
  };
use onnxruntime::session::Session;
use tracing_subscriber::FmtSubscriber;
use tracing::Level;
use ndarray::Array;
use std::sync::Arc;
use std::sync::Mutex;

// tract crate
// use tract_ndarray::Array;
use tract_onnx::prelude::*;

type Error = Box<dyn std::error::Error>;

#[derive(Debug)]
pub struct OnnxSession {
    pub model_filepath: String,
    pub environment: Environment,
    // pub session: Arc<Session>
    // pub input_shape: Vec<usize>,
    // pub output_shape: Vec<usize>
}


impl OnnxSession  {
    pub fn new(model_filepath: String) -> Result<OnnxSession, Error>  {

        let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();

        let environment = Environment::builder()
        .with_name("test_environment")
        // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
        .with_log_level(LoggingLevel::Info)
        .build()?;

        let mut session = environment
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::Basic)?
            .with_number_threads(1)?
            .with_model_from_file(model_filepath.clone())?;

        tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

        Ok(
            OnnxSession {
                model_filepath: model_filepath.to_owned(),
                environment,
                // session: Arc::new(session),
                // input_shape,
                // output_shape
            }
        )
    }

    // pub fn predict(mut self, input: Array1<f32>)  /* -> bool */ {
    //     let mut input_shape = self.input_shape;
    //     // let output_shape = self.output_shape;
    //     // total input dimensions
    //     let mut n = 1;
    //     for el in input_shape.iter_mut() {
    //         n *= *el;
    //     }
    //     dbg!("total input dims: {}", n);
    //
    //     // let array = input.into_shape(input_shape).unwrap();
    //
    //     let array = Array::linspace(0.0_f32, 1.0, n as usize)
    //     .into_shape(input_shape)
    //     .unwrap();
    //     // println!("input array: {:?}", array);
    //     let input_tensor_values = vec![array];
    //     println!("input_tensor_values: {:?}", input_tensor_values);
    //     let predictions: Vec<OrtOwnedTensor<f32, _>> = self.session.run(input_tensor_values).unwrap();
    //     println!("predictions: {:?}", predictions);
    //     let result: Vec<_> = predictions.iter().map(|el| el.to_owned()).collect();
    //     println!("result: {:?}", result);
    //     // true
    // }

    pub fn run(&self, sample: Vec<f32>) -> Result<Vec<f32>, Error> {
        // tracing::subscriber::set_global_default(self.subscriber).expect("setting default subscriber failed");

        let mut session = self.environment
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::Basic)?
            .with_number_threads(1)?
            .with_model_from_file(self.model_filepath.clone())?;

            // let session = self.session.clone();
        // let mut session = self.session.lock().unwrap();

            let mut input_shape: Vec<usize> = session.inputs[0]
                .dimensions()
                .map(|d| {
                    let curdim = match d {
                        Some(dim) => dim,
                        None => 1
                    };
                    curdim
                })
                .collect();

            println!("input_0: {:?}", input_shape);

            let output_shape: Vec<usize> = session.outputs[0]
                .dimensions()
                .map(|d| {
                    let curdim = match d {
                        Some(dim) => dim,
                        None => 1
                    };
                    curdim
                    // d.unwrap()
                })
                .collect();
            println!("output_0: {:?}", output_shape);

            // TODO if dimensions present in toml config, assert it matches
            // assert_eq!(input0_shape, [1, 3, 224, 224]);
            // assert_eq!(output0_shape, [1, 1000, 1, 1]);

            // total input dimensions
            let mut n = 1;
            for el in input_shape.iter_mut() {
                n *= *el;
            }
            dbg!("total input dims: {}", n);

            // Prepare sample and infer into runtime session
            let sample = Array::from(sample)
                        .into_shape(input_shape)
                        .unwrap();
            let sample = vec![sample];
            let predictions: Vec<OrtOwnedTensor<f32, _>> = session.run(sample).unwrap();

            // Format predictions and return
            let mut preds: Vec<f32> = vec![];
            for tensor in  predictions {
                for pred in tensor.iter() {
                    preds.push(*pred);
                }
            }
        Ok(preds)
    }
}

// fn load_model() -> TractResult<()> {
//     let model_filename = "simple_model.onnx";
//     let model = tract_onnx::onnx()
//     // load the model
//     .model_for_path(model_filename)?
//     // specify input type and shape
//     .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224, 224)))?
//     // optimize the model
//     .into_optimized()?
//     // make the model runnable and fix its inputs and outputs
//     .into_runnable()?;
//     // let some = model.parse();
//     Ok(())
// }

// pub fn create_runtime(model_filepath: String) -> Result<OnnxSession, Error> {
//     let subscriber = FmtSubscriber::builder()
//         .with_max_level(Level::TRACE)
//         .finish();
//     let environment = Environment::builder()
//     .with_name("test_environment")
//     // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
//     .with_log_level(LoggingLevel::Info)
//     .build()?;
//
//     tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
//
//     let mut session = environment
//         .new_session_builder()?
//         .with_optimization_level(GraphOptimizationLevel::Basic)?
//         .with_number_threads(1)?
//         .with_model_from_file(model_filepath)?;
//
//         let mut input_shape: Vec<usize> = session.inputs[0]
//         .dimensions()
//         .map(|d| {
//             let curdim = match d {
//                 Some(dim) => dim,
//                 None => 1
//             };
//             curdim
//         })
//         .collect();
//     println!("input_0: {:?}", input_shape);
//
//     let mut output_shape: Vec<usize> = session.outputs[0]
//         .dimensions()
//         .map(|d| {
//             let curdim = match d {
//                 Some(dim) => dim,
//                 None => 1
//             };
//             curdim
//             // d.unwrap()
//         })
//         .collect();
//     println!("output_0: {:?}", output_shape);
//
//     // TODO if dimensions present in toml config, assert it matches
//     // assert_eq!(input0_shape, [1, 3, 224, 224]);
//     // assert_eq!(output0_shape, [1, 1000, 1, 1]);
//
//     let onnx_session = OnnxSession {
//         session,
//         input_shape,
//         output_shape
//     };
//     Ok(onnx_session)
// }

// pub fn predict_in_runtime(mut session: OnnxSession, input: Array1<f32>) {
//     let mut input_shape = session.input_shape;
//     let output_shape = session.output_shape;
//     // total input dimensions
//     let mut n = 1;
//     for el in input_shape.iter_mut() {
//         n *= *el;
//     }
//     dbg!("total input dims: {}", n);
//
//     // let array = input.into_shape(input_shape).unwrap();
//
//     let array = Array::linspace(0.0_f32, 1.0, n as usize)
//     .into_shape(input_shape)
//     .unwrap();
//     // println!("input array: {:?}", array);
//
//     let input_tensor_values = vec![array];
//     println!("input_tensor_values: {:?}", input_tensor_values);
//
//     let predictions: Vec<OrtOwnedTensor<f32, _>> = session.session.run(input_tensor_values).unwrap();
//     println!("predictions: {:?}", predictions);
// }

// pub fn run(model_filepath: String) -> Result<(), Error> {
//     let subscriber = FmtSubscriber::builder()
//         .with_max_level(Level::TRACE)
//         .finish();
//     let environment = Environment::builder()
//     .with_name("test_environment")
//     // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
//     .with_log_level(LoggingLevel::Info)
//     .build()?;
//
//     tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
//     let mut session = environment
//         .new_session_builder()?
//         .with_optimization_level(GraphOptimizationLevel::Basic)?
//         .with_number_threads(1)?
//         .with_model_from_file(model_filepath)?;
//
//         // dbg!(&session);
//         // let inputs = &session.inputs;
//         // let outputs = &session.outputs;
//         //
//         // for input in inputs {
//         //     // dbg!(&input);
//         //     let in_dim: Vec<Option<usize>> = input.dimensions().map(|d| d).collect();
//         //     println!("input_dims: {:?}", in_dim);
//         // }
//         //
//         // for output in outputs {
//         //     // dbg!(&output);
//         //     let out_dim: Vec<Option<usize>> = output
//         //     .dimensions()
//         //     .map(|d| d)
//         //     .collect();
//         //     println!("output_dims: {:?}", out_dim);
//         // }
//
//         let mut input0_shape: Vec<usize> = session.inputs[0]
//             .dimensions()
//             .map(|d| {
//                 let curdim = match d {
//                     Some(dim) => dim,
//                     None => 1
//                 };
//                 curdim
//             })
//             .collect();
//         println!("input_0: {:?}", input0_shape);
//         let mut output0_shape: Vec<usize> = session.outputs[0]
//             .dimensions()
//             .map(|d| {
//                 let curdim = match d {
//                     Some(dim) => dim,
//                     None => 1
//                 };
//                 curdim
//                 // d.unwrap()
//             })
//             .collect();
//         println!("output_0: {:?}", output0_shape);
//
//         // TODO if dimensions present in toml config, assert it matches
//         // assert_eq!(input0_shape, [1, 3, 224, 224]);
//         // assert_eq!(output0_shape, [1, 1000, 1, 1]);
//
//         // total input dimensions
//         let mut n = 1;
//         for el in input0_shape.iter_mut() {
//             n *= *el;
//         }
//         dbg!("total input dims: {}", n);
//
//         let array = Array::linspace(0.0_f32, 1.0, n as usize)
//         .into_shape(input0_shape)
//         .unwrap();
//         // println!("input array: {:?}", array);
//
//         let input_tensor_values = vec![array];
//         println!("input_tensor_values: {:?}", input_tensor_values);
//
//         let predictions: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values).unwrap();
//         println!("predictions: {:?}", predictions);
//
//         // let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values)?;
//         // dbg!("output: {} len: {}", &outputs, &outputs.len());
//         // println!("{:?}", outputs);
//
//     Ok(())
// }
