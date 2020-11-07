mod model;
mod routes;
mod configuration;
use onnxruntime::{environment::Environment,
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel,
    LoggingLevel, session
  };

use std::*;
use std::sync::Arc;
use actix_web::http::{header, Method, StatusCode};
use actix_web::{get, post, web, App, HttpServer, Responder, HttpResponse, HttpRequest, error, Error};
use serde::{Deserialize, Serialize};
use ndarray::{Array, ArrayBase, Array1} ;
use crate::model::onnx::{ OnnxSession };
use crate::routes::{status, predict};
use crate::configuration::get_configuration_from_file;

// #[get("/status")]
// async fn index() -> impl Responder {
//     format!("Status ok!")
// }
//
// /// This handler manually load request payload and parse json object
// async fn index_manual(mut payload: web::Payload) -> Result<HttpResponse, Error> {
//     // payload is a stream of Bytes objects
//     let mut body = web::BytesMut::new();
//
//     // while let Some(chunk) = payload.next().await {
//     //     let chunk = chunk?;
//     //     // limit max size of in-memory payload
//     //     if (body.len() + chunk.len()) > MAX_SIZE {
//     //         return Err(error::ErrorBadRequest("overflow"));
//     //     }
//     //     body.extend_from_slice(&chunk);
//     // }
//
//     // body is loaded, now we can deserialize serde-json
//     let obj = serde_json::from_slice::<MyObj>(&body)?;
//     Ok(HttpResponse::Ok().json(obj)) // <- send response
// }


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = get_configuration_from_file("sample.toml").unwrap();
    let _model_format = config.model.format;
    let model_path = config.model.path;
    let host = config.api.host;
    let port = config.api.port;
    let predict_endpoint = config.api.predict_endpoint;
    // let base_endpoint = config.api.base_endpoint;
    // let input_dims = config.model.input_dims;
    // let output_dims = config.model.output_dims;

    println!("Loading model and preparing runtime... ");

    // TODO
    // check model_format is "onnx"

    // TODO move this into endpoint
    // if let Err(e) = run(model_path.clone()) {
    //     println!("Encountered an error {}. Exiting...", e);
    //     std::process::exit(1);
    // }

    let onnx_session = OnnxSession::new(model_path.clone()).unwrap();
    let input_sample: Array1<f32> = Array1::from(vec![1.,2.,3.]);
    let preds = onnx_session.run();
    println!("preds: {:?}", preds);

    // let _ = onnx_session.predict(input_sample);
    // let onnx_session = create_onnx_session(model_path);
    // onnx_inference(onnx_session, input_sample);
    // let _ = run(model_path.clone());



    println!("Launching web server");

    HttpServer::new(move ||
        App::new()
        .service(status)
        // .service(
        //     web::resource("/predict").to(|req: HttpRequest| match *req.method() {
        //         Method::GET => HttpResponse::MethodNotAllowed(),
        //         Method::POST => HttpResponse::Ok(),
        //         _ => HttpResponse::NotFound(),
        //     }),
        // )
        .service(
            web::resource(format!("{}", predict_endpoint.clone()))
            .route(web::post()
            .to(predict)))

    )
    .bind(format!("{}:{}", host, port))?
    .run()
    .await

}
