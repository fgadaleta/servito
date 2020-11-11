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

// pub async fn predict_helper(payload: web::Bytes) -> Result<HttpResponse, Error> {
//     // payload is a stream of Bytes objects
//     // let mut body = web::BytesMut::new();
//     let body = std::str::from_utf8(&payload).unwrap();
//     // println!("payload: {:?}", body);
//     let obj = serde_json::from_slice::<Payload>(body.as_bytes())?;
//     // println!("obj: {:?}", obj.input);
//     let input_vector: Vec<f64> = obj.input
//     .split(",")
//     .map(|s| s.parse().expect("parse error"))
//     .collect();
//     println!("input_vector: {:?}", input_vector);
//     // TODO deserialize body into a vector of floats
//     // body is loaded, now we can deserialize serde-json
//     Ok(HttpResponse::Ok().json(obj)) // <- send response
// }


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = get_configuration_from_file("sample.toml").unwrap();
    let model_format = config.model.format;
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
    //
    // TODO move this into endpoint
    // if let Err(e) = run(model_path.clone()) {
    //     println!("Encountered an error {}. Exiting...", e);
    //     std::process::exit(1);
    // }

    let onnx_session = Arc::new(OnnxSession::new(model_path.clone()).unwrap());
    // let input_sample: Array1<f32> = Array1::from(vec![1.,2.,3.]);
    // let preds = onnx_session.run();
    // println!("preds: {:?}", preds);

    println!("Launching web server");

    HttpServer::new(move | | {
        // { let onnx_session = onnx_session.clone(); }
        let onnx_session = onnx_session.clone();

        App::new()
        .service(status)
        .service(
            web::resource("/predict").to(move |req: HttpRequest| match *req.method() {
                Method::GET => HttpResponse::MethodNotAllowed(),
                Method::POST => {
                    // let onnx_session = onnx_session.clone();
                    let preds = onnx_session.run();
                    println!("preds: {:?}", preds);
                    HttpResponse::Ok()
                },
                _ => HttpResponse::NotFound(),
            }),
        )
        // .service(
        // web::resource(format!("{}", predict_endpoint.clone()))
        //         .route(web::post().to(predict)))
    })
    .bind(format!("{}:{}", host, port))?
    .run()
    .await

}
