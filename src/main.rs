mod model;
mod routes;
mod configuration;
// use onnxruntime::{environment::Environment,
//     tensor::OrtOwnedTensor,
//     GraphOptimizationLevel,
//     LoggingLevel, session
//   };
use ndarray::{Array, Array1} ;

use std::*;
use std::sync::Arc;
use actix_web::http::{header, Method, StatusCode};
use actix_web::{get, post, web, middleware, App, HttpServer, Responder, HttpResponse, HttpRequest, error, Error, Result};
use serde::{Deserialize, Serialize};
use json::JsonValue;
// use ndarray::{Array, ArrayBase, Array1} ;
use crate::model::onnx::OnnxSession;
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


enum SessionType {
    ONNX,
    TORCH,
    // TODO
}

struct RuntimeSession<T> {
    session: Arc<T>,
    session_type: SessionType
}


#[derive(Debug, Serialize, Deserialize)]
struct Payload {
    input: Vec<f32>,
}

// #[derive(Debug)]
// struct AppState {
//     foo: String,
//     onnx_session: Arc<OnnxSession>,
// }

/// extract `Info` using serde
// async fn index(info: web::Json<Info>) -> Result<String> {
//     dbg!("Payload input ", &info.input);
//     Ok(format!("Payload input {:?}!", info.input))
// }

/// This handler uses json extractor
// async fn predict_handler(state: web::Data<AppState>, body: web::Bytes) -> Result<HttpResponse, Error> {
//     // dbg!("payload: {:?}", &item);
//     // HttpResponse::Ok().json(item.input.clone()) // <- send response
//     // body is loaded, now we can deserialize json-rust
//     let result = json::parse(std::str::from_utf8(&body).unwrap()); // return Result
//         let injson: JsonValue = match result {
//             Ok(v) => {
//                 dbg!("state: {:?} ", &state);
//                 dbg!("body: {:?} ", &v);
//                 v
//             },
//             Err(e) => json::object! {"err" => e.to_string() },
//         };
//         Ok(HttpResponse::Ok()
//             .content_type("application/json")
//             .body(injson.dump()))
// }
// async fn extract_item(req: HttpRequest) -> HttpResponse {
//     dbg!("request: {:?}", req);
//     // println!("model: {:?}", item);
//     HttpResponse::Found().finish()
// }

/// This handler uses json extractor
async fn index(item: web::Json<Payload>) -> HttpResponse {
    println!("model: {:?}", &item);
    HttpResponse::Ok().json(item.input.clone()) // <- send response
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
    // let input_dims = config.model.input_dims;
    // let output_dims = config.model.output_dims;

    std::env::set_var("RUST_LOG", "actix_web=info");
    env_logger::init();

    println!("Loading model and preparing runtime... ");

    // TODO
    // check model_format is "onnx"
    //
    // TODO move this into endpoint
    // if let Err(e) = run(model_path.clone()) {
    //     println!("Encountered an error {}. Exiting...", e);
    //     std::process::exit(1);
    // }

    // let runtime_session = Arc::new(RuntimeSession);

    // if model_format == "onnx" {
    // TODO encapsulate into a RuntimeSession struct with type (Arc:new(..))
    let onnx_session = Arc::new(OnnxSession::new(model_path.clone()).unwrap());
    // }

    println!("Launching web server");

    HttpServer::new(move | | {
        let onnx_session = onnx_session.clone();

        App::new()
        .service(status)
        .service(web::resource("/predict2").route(web::post().to(index)))
        .service(
            web::resource("/predict").to(move |payload: web::Json<Payload>, req: HttpRequest |
                match *req.method() {

                    Method::GET => HttpResponse::MethodNotAllowed().finish(),

                    Method::POST => {
                        let preds = onnx_session.run(payload.input.clone());
                        // dbg!("payload input: ", &payload.input);
                        // dbg!("preds: {:?}", &preds);
                        HttpResponse::Ok().json(preds.unwrap())
                    },
                    _ => HttpResponse::NotFound().finish(),
            }),
        )
    })
    .bind(format!("{}:{}", host, port))?
    .run()
    .await

    /*
    // kinda working
    HttpServer::new(move | | {
        dbg!("i am here");
        let onnx_session = onnx_session.clone();

        App::new()
        .service(status)

        .service(
            web::resource("/predict")
            .data(web::JsonConfig::default().limit(1024))
            .to(move |item: web::Json<Info>, req: HttpRequest |
                {
                    println!("and now i am here");

                match *req.method() {

                    Method::GET => HttpResponse::MethodNotAllowed(),

                    Method::POST => {
                        // TODO get input data from payload
                        println!("hello i am here");

                        dbg!("req: ", &req);
                        dbg!("item: {:?}", &item);

                        let headers = req.headers();
                        dbg!("headers: ", headers);
                        // let input_sample: Array1<f32> = Array1::from(vec![1.,2.,3.]);
                        let preds = onnx_session.run();

                        println!("preds: {:?}", preds);
                        HttpResponse::Ok()
                    },

                    _ => HttpResponse::NotFound(),
                }
                }
            ),
        )

        .service(
            web::resource("/predict2")
                .data(web::JsonConfig::default().limit(1024)) // <- limit size of the payload (resource level)
                .route(web::post()
                .to(extract_item)),
        )

    })
    .bind(format!("{}:{}", host, port))?
    .run()
    .await
    */

}
