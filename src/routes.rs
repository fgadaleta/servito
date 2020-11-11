use actix_web::http::{header, Method, StatusCode};
use actix_web::{get, post, web, App, HttpServer, Responder, HttpResponse, HttpRequest, error, Error};
// use json::JsonValue;
use serde::{Deserialize, Serialize};
// use onnxruntime::session::Session;

// const MAX_SIZE: usize = 262_144; // max payload size is 256k

#[derive(Debug, Serialize, Deserialize)]
pub struct Payload<'a>  {
    pub input: &'a str,
    // _meta: u32,
}
// impl Payload

struct ModelInput {
    data: Vec<f64>
}

// TODO define input data type from model specs


#[get("/status")]
pub async fn status() -> impl Responder {
    format!("Status ok!")
}


/// This handler manually load request payload and parse json object
pub async fn predict(payload: web::Bytes) -> Result<HttpResponse, Error> {
    // payload is a stream of Bytes objects
    // let mut body = web::BytesMut::new();
    let body = std::str::from_utf8(&payload).unwrap();
    // println!("payload: {:?}", body);
    let obj = serde_json::from_slice::<Payload>(body.as_bytes())?;
    // println!("obj: {:?}", obj.input);
    let input_vector: Vec<f64> = obj.input
    .split(",")
    .map(|s| s.parse().expect("parse error"))
    .collect();
    println!("input_vector: {:?}", input_vector);


    // TODO deserialize body into a vector of floats

    // body is loaded, now we can deserialize serde-json

    Ok(HttpResponse::Ok().json(obj)) // <- send response
}
