use tract_onnx::prelude::*;
use onnxruntime::{environment::Environment,
                tensor::OrtOwnedTensor,
                GraphOptimizationLevel,
                LoggingLevel,
                 };
use ndarray::Array;
use tracing::Level;
use std::*;
use tracing_subscriber::FmtSubscriber;

type Error = Box<dyn std::error::Error>;


fn load_model() -> TractResult<()> {
    let model_filename = "super_resolution.onnx";

    let model = tract_onnx::onnx()
    // load the model
    .model_for_path(model_filename)?
    // specify input type and shape
    .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224, 224)))?
    // optimize the model
    .into_optimized()?
    // make the model runnable and fix its inputs and outputs
    .into_runnable()?;
    // let some = model.parse();

    Ok(())
}


fn run() -> Result<(), Error> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();

    let environment = Environment::builder()
    .with_name("test_environment")
    // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
    .with_log_level(LoggingLevel::Warning)
    .build()?;

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        // NOTE: The example uses SqueezeNet 1.0 (ONNX version: 1.3, Opset version: 8),
        //       _not_ SqueezeNet 1.1 as downloaded by '.with_model_downloaded(ImageClassification::SqueezeNet)'
        //       Obtain it with:
        //          curl -LO "https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-8.onnx"
        .with_model_from_file("super_resolution.onnx")?;

        dbg!(&session);
        // let inputs = &session.inputs;
        // let outputs = &session.outputs;
        //
        // for input in inputs {
        //     // dbg!(&input);
        //     let in_dim: Vec<Option<usize>> = input.dimensions().map(|d| d).collect();
        //     println!("input_dims: {:?}", in_dim);
        // }
        //
        // for output in outputs {
        //     // dbg!(&output);
        //     let out_dim: Vec<Option<usize>> = output
        //     .dimensions()
        //     .map(|d| d)
        //     .collect();
        //     println!("output_dims: {:?}", out_dim);
        // }

        let mut input0_shape: Vec<usize> = session.inputs[0]
            .dimensions()
            .map(|d| {
                let curdim = match d {
                    Some(dim) => dim,
                    None => 0
                };
                curdim
                // d.unwrap()
            })
            .collect();

        println!("input_0: {:?}", input0_shape);

        let output0_shape: Vec<usize> = session.outputs[0]
            .dimensions()
            .map(|d| {
                let curdim = match d {
                    Some(dim) => dim,
                    None => 0
                };
                curdim
                // d.unwrap()
            })
            .collect();
        println!("output_0: {:?}", output0_shape);

        // TODO if dimensions present in toml config, assert it matches
        // assert_eq!(input0_shape, [1, 3, 224, 224]);
        // assert_eq!(output0_shape, [1, 1000, 1, 1]);

        // total input dimensions
        let mut nonzero_input: Vec<usize> = input0_shape.clone()
        .into_iter()
        .filter(|&i| i > 0)
        .collect();

        let mut n = 1;
        for el in nonzero_input.iter_mut() {
            n *= *el;
        }
        dbg!("total input dims: {}", n);

        let array = Array::linspace(0.0_f32, 1.0, n as usize)
        .into_shape(nonzero_input)
        .unwrap();
        let input_tensor_values = vec![array];
        // println!("{:?}", input_tensor_values);
        let predictions: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values).unwrap();

        // let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values)?;
        // dbg!("output: {} len: {}", &outputs, &outputs.len());
        // println!("{:?}", outputs);

    Ok(())
}


fn main() {
    println!("Loading model ");

    if let Err(e) = run() {
        println!("Encountered an error {}. Exiting...", e);
        std::process::exit(1);
    }

    // let model = load_model();
    // dbg!(model);



}
