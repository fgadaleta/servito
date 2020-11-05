mod model;
mod routes;

use std::*;
use actix_web::http::{header, Method, StatusCode};
use actix_web::{get, post, web, App, HttpServer, Responder, HttpResponse, HttpRequest, error, Error};
// use json::JsonValue;
use serde::{Deserialize, Serialize};
// use serde_json::*;
use crate::model::onnx::run;
use crate::routes::{status, predict};
// type Error = Box<dyn std::error::Error>;
const MAX_SIZE: usize = 262_144; // max payload size is 256k




// #[get("/status")]
// async fn index() -> impl Responder {
//     format!("Status ok!")
// }


// /// This handler manually load request payload and parse json object
// async fn index_manual(mut payload: web::Payload) -> Result<HttpResponse, Error> {
//     // payload is a stream of Bytes objects
//     let mut body = web::BytesMut::new();

//     // while let Some(chunk) = payload.next().await {
//     //     let chunk = chunk?;
//     //     // limit max size of in-memory payload
//     //     if (body.len() + chunk.len()) > MAX_SIZE {
//     //         return Err(error::ErrorBadRequest("overflow"));
//     //     }
//     //     body.extend_from_slice(&chunk);
//     // }

//     // body is loaded, now we can deserialize serde-json
//     let obj = serde_json::from_slice::<MyObj>(&body)?;
//     Ok(HttpResponse::Ok().json(obj)) // <- send response
// }

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
        .service(status)
        // .service(
        //     web::resource("/predict").to(|req: HttpRequest| match *req.method() {
        //         Method::GET => HttpResponse::MethodNotAllowed(),
        //         Method::POST => HttpResponse::Ok(),
        //         _ => HttpResponse::NotFound(),
        //     }),
        // )
        .service(
            web::resource("/predict").route(web::post().to(predict)))

    )
    .bind("127.0.0.1:6666")?
    .run()
    .await

}
