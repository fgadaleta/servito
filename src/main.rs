mod configuration;
mod model;
mod routes;
use actix_web::http::Method;
use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::*;

// use tract_onnx::prelude::*;
use crate::configuration::get_configuration_from_file;
use crate::model::onnx::OnnxSession;
use crate::routes::status;

type SessionError = Box<dyn std::error::Error>;


#[derive(Debug, Serialize, Deserialize)]
struct Payload {
    input: Vec<f32>,
}

/// This handler uses json extractor
///
async fn handler_train(item: web::Json<Payload>) -> HttpResponse {
    unimplemented!();
    // HttpResponse::Ok().json(item.input.clone()) // <- send response
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = get_configuration_from_file("sample.toml").unwrap();
    // let model_format = config.model.format;
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

    // use onnx runtime
    let onnx_session = Arc::new(OnnxSession::new(model_path.clone()).unwrap());

    println!("Launching web server");

    HttpServer::new(move || {
        let onnx_session = onnx_session.clone(); // cloning a ref
                                                 // let runtime_session = runtime_session.clone();
                                                 // let torch_session = torch_session.clone();

        App::new()
            .data(web::JsonConfig::default().limit(4096)) // <- limit size of the payload (global configuration)
            .service(status)
            .service(web::resource("/train").route(web::post().to(handler_train)))
            .service(web::resource("/predict").to(
                move |payload: web::Json<Payload>, req: HttpRequest| match *req.method() {

                    Method::GET => HttpResponse::MethodNotAllowed().finish(),

                    Method::POST => {
                        let sample = payload.input.clone();
                        let preds = onnx_session.run(sample.clone());
                        dbg!("preds: {:?}", &preds);
                        HttpResponse::Ok().json(preds.unwrap())
                    }
                    _ => HttpResponse::NotFound().finish(),
                },
            ))
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
