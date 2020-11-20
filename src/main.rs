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
use crate::model::onnx::OnnxSession;
use crate::routes::{status, predict};
use crate::configuration::get_configuration_from_file;


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
    let input_dims = config.model.input_dims;
    // let output_dims = config.model.output_dims;

    std::env::set_var("RUST_LOG", "actix_web=info");
    env_logger::init();

    println!("Loading model and preparing runtime... ");

    // TODO
    // check model_format is "onnx"

    // let runtime_session = Arc::new(RuntimeSession);

    if model_format == "onnx" {

    }

        // TODO encapsulate into a RuntimeSession struct with type (Arc:new(..))
    let onnx_session = Arc::new(OnnxSession::new(model_path.clone()).unwrap());
    // }

    println!("Launching web server");

    HttpServer::new(move | | {
        let onnx_session = onnx_session.clone();

        App::new()
        .data(web::JsonConfig::default().limit(4096)) // <- limit size of the payload (global configuration)
        .service(status)
        // .service(web::resource("/predict2").route(web::post().to(index)))
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

}



#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::dev::Service;
    use actix_web::{http, test, web, App};

    async fn test_index() -> Result<(), Error> {
        let mut app = test::init_service(
            App::new().service(web::resource("/").route(web::post().to(index))),
        )
        .await;

        let req = test::TestRequest::post()
            .uri("/")
            .set_json(&Payload {
                input: vec![43.0,1.0,2.0,3.0],
            })
            .to_request();
        let resp = app.call(req).await.unwrap();

        assert_eq!(resp.status(), http::StatusCode::OK);

        let response_body = match resp.response().body().as_ref() {
            Some(actix_web::body::Body::Bytes(bytes)) => bytes,
            _ => panic!("Response error"),
        };

        assert_eq!(response_body, r##"{"name":"my-name","number":43}"##);

        Ok(())
    }
}