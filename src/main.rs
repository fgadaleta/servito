mod model;
mod routes;
mod configuration;
use ndarray::{Array, Array1} ;
use std::*;
use std::sync::{Arc, Mutex};
use actix_web::http::{header, Method, StatusCode};
use actix_web::{get, post, web, middleware, App, HttpServer, Responder, HttpResponse, HttpRequest, error, Error, Result};
use serde::{Deserialize, Serialize};
use json::JsonValue;

// use tract_ndarray::Array;
use tract_onnx::prelude::*;

use crate::model::onnx::OnnxSession;
use crate::model::torch::TorchSession;
use crate::routes::{status, predict};
use crate::configuration::get_configuration_from_file;

type SessionError = Box<dyn std::error::Error>;


#[derive(Debug, Clone)]
enum SessionType {
    ONNX,
    TORCH,
    // TODO
}

#[derive(Debug)]
struct RuntimeSession<T> {
    session: Arc<T>,
    session_type: SessionType
}

#[derive(Debug, Clone)]
enum RuntimeSession2 {
    ONNX { session: Arc<OnnxSession> },
    TORCH { session: Arc<TorchSession> }
}

impl RuntimeSession2 {
    fn run(&self, sample: Vec<f32>) -> Result<Vec<f32>, SessionError> {
    match self {
        Self::ONNX { session } => session.run(sample),
        // Self::TORCH { session } => session.run(sample),
        _ => unimplemented!()
        }
    }
}

impl<T> RuntimeSession<T> {
    fn new(session: Arc<T>, session_type: SessionType) -> Self {

        RuntimeSession{
            session,
            session_type
        }
    }
}

impl<T> Clone for RuntimeSession<T> {
    fn clone(&self) -> Self {
        Self{
            session: self.session.clone(),
            session_type: self.session_type.clone()
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Payload {
    input: Vec<f32>,
}


/// This handler uses json extractor
async fn handler_train(item: web::Json<Payload>) -> HttpResponse {

    // println!("model: {:?}", &item);
    // TODO
    unimplemented!();

    // HttpResponse::Ok().json(item.input.clone()) // <- send response
}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = get_configuration_from_file("sample.toml").unwrap();
    let model_format = config.model.format;
    let model_path = config.model.path;
    let host = config.api.host;
    let port = config.api.port;
    // let predict_endpoint = config.api.predict_endpoint;
    // let base_endpoint = config.api.base_endpoint;
    let input_dims = config.model.input_dims;
    let output_dims = config.model.output_dims;

    std::env::set_var("RUST_LOG", "actix_web=info");
    env_logger::init();

    dbg!("input_dims: ", &input_dims);
    dbg!("output_dims: ", &output_dims);

    println!("Loading model and preparing runtime... ");

    // TODO
    // check model_format is "onnx"
    // let session_type = match model_format {

    // }
    // if model_format == "onnx" {
    // }

    // use onnx runtime
    let onnx_session = Arc::new(OnnxSession::new(model_path.clone()).unwrap());

    // use tract runtime
    let tract_runtime = tract_onnx::onnx()
        .model_for_path(model_path.clone()).unwrap()
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1, 34))).unwrap()
        .into_optimized().unwrap()
        .into_runnable().unwrap();


    // use torch runtime
    let torchscript_path = "/Users/frag/Documents/projects/amethix-ncode/servito/python/test.pt";
    // let torch_session = Arc::new(TorchSession::new(&model_path[..]));
    // let torch_session = TorchSession::new(torchscript_path);

    // let model = tch::CModule::load(torchscript_path).unwrap();
    // let runtime_session = RuntimeSession::new(onnx_session.clone(), SessionType::ONNX);

    println!("Launching web server");

    HttpServer::new(move || {
        let onnx_session = onnx_session.clone();  // cloning a ref
        // let runtime_session = runtime_session.clone();
        // let torch_session = torch_session.clone();

        App::new()
        .data(web::JsonConfig::default().limit(4096)) // <- limit size of the payload (global configuration)
        .service(status)
        .service(web::resource("/train").route(web::post().to(handler_train)))
        .service(
            web::resource("/predict").to(move |payload: web::Json<Payload>, req: HttpRequest |
                match *req.method() {

                    Method::GET => HttpResponse::MethodNotAllowed().finish(),

                    Method::POST => {
                        // let session = runtime_session.session_type;
                        // let preds = runtime_session.session.run(payload.input.clone());
                        let sample = payload.input.clone();
                        let preds = onnx_session.run(sample.clone());

                        // run sample through tract runtime
                        // let image: Tensor = Array::from_shape_fn((1, 1, 34).into();
                        // let result = tract_runtime.run(tvec!(sample)).unwrap();

                        // let preds = torch_session.run(sample);

                        // dbg!("payload input: ", &payload.input);
                        dbg!("preds: {:?}", &preds);
                        HttpResponse::Ok().json(preds.unwrap())
                    },
                    _ => HttpResponse::NotFound().finish(),
            }),
        )
    })
    .bind(format!("{}:{}", host, port))?
    .run()
    .await
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use actix_web::dev::Service;
//     use actix_web::{http, test, web, App};
//
//     async fn test_index() -> Result<(), Error> {
//         let mut app = test::init_service(
//             App::new().service(web::resource("/").route(web::post().to(index))),
//         )
//         .await;
//
//         let req = test::TestRequest::post()
//             .uri("/")
//             .set_json(&Payload {
//                 input: vec![43.0,1.0,2.0,3.0],
//             })
//             .to_request();
//         let resp = app.call(req).await.unwrap();
//
//         assert_eq!(resp.status(), http::StatusCode::OK);
//
//         let response_body = match resp.response().body().as_ref() {
//             Some(actix_web::body::Body::Bytes(bytes)) => bytes,
//             _ => panic!("Response error"),
//         };
//
//         assert_eq!(response_body, r##"{"name":"my-name","number":43}"##);
//
//         Ok(())
//     }
// }
