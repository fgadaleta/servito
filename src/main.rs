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
use actix_web::http::{header, Method, StatusCode};
use actix_web::{get, post, web, App, HttpServer, Responder, HttpResponse, HttpRequest};
type Error = Box<dyn std::error::Error>;


fn load_model() -> TractResult<()> {
    let model_filename = "simple_model.onnx";

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
    .with_log_level(LoggingLevel::Info)
    .build()?;

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .with_model_from_file("simple_model.onnx")?;

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
                    None => 1
                };
                curdim
            })
            .collect();

        println!("input_0: {:?}", input0_shape);

        let mut output0_shape: Vec<usize> = session.outputs[0]
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
        println!("output_0: {:?}", output0_shape);

        // TODO if dimensions present in toml config, assert it matches
        // assert_eq!(input0_shape, [1, 3, 224, 224]);
        // assert_eq!(output0_shape, [1, 1000, 1, 1]);

        // total input dimensions
        let mut n = 1;
        for el in input0_shape.iter_mut() {
            n *= *el;
        }
        dbg!("total input dims: {}", n);

        let array = Array::linspace(0.0_f32, 1.0, n as usize)
        .into_shape(input0_shape)
        .unwrap();
        // println!("input array: {:?}", array);

        let input_tensor_values = vec![array];
        println!("input_tensor_values: {:?}", input_tensor_values);

        let predictions: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values).unwrap();
        println!("predictions: {:?}", predictions);

        // let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values)?;
        // dbg!("output: {} len: {}", &outputs, &outputs.len());
        // println!("{:?}", outputs);

    Ok(())
}

#[get("/status")]
async fn index() -> impl Responder {
    format!("Status ok!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Loading model and preparing runtime... ");

    if let Err(e) = run() {
        println!("Encountered an error {}. Exiting...", e);
        std::process::exit(1);
    }

    println!("Launching web server");

    HttpServer::new(||
        App::new()
        .service(index)
        .service(
            web::resource("/predict").to(|req: HttpRequest| match *req.method() {
                Method::GET => HttpResponse::MethodNotAllowed(),
                Method::POST => HttpResponse::Ok(),
                _ => HttpResponse::NotFound(),
            }),
        )
    )
    .bind("127.0.0.1:6666")?
    .run()
    .await

}
