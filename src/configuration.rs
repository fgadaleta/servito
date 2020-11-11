use std::fs::File;
use std::io::prelude::*;
use std::io::Error;
use toml;
use serde_derive::Deserialize;


#[derive(Deserialize, Debug)]
pub struct Config {
    pub api: Api,
    pub model: Model,
    pub settings: Settings

}

#[derive(Deserialize, Debug)]
pub struct Api {
    pub host: String,
    pub port: u16,
    pub base_endpoint: String,
    pub predict_endpoint: String,
    pub train_endpoint: String,
    pub test_endpoint: String
}

#[derive(Deserialize, Debug)]
pub struct Model {
    pub path: String,
    pub format: String,
    pub input_dims: String,
    pub output_dims: String
}

#[derive(Deserialize, Debug)]
pub struct Settings {
    pub something: String,
    // TODO other settings here
}

pub fn get_content_from_file(filepath: &str) -> Result<String, Error> {
    let mut file = File::open(filepath)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    Ok(content)
}

pub fn get_configuration_from_file(filepath: &str) -> Result<Config, Error> {
    let configuration = get_content_from_file(filepath)?;
    let config: Config = toml::from_str(configuration.as_str()).unwrap();

    Ok(
        Config {
            api: Api {
                host: config.api.host,
                port: config.api.port,
                base_endpoint: config.api.base_endpoint,
                predict_endpoint: config.api.predict_endpoint,
                train_endpoint: config.api.train_endpoint,
                test_endpoint: config.api.test_endpoint,
            },

            model: Model {
                path: config.model.path,
                format: config.model.format,
                input_dims: config.model.input_dims,
                output_dims: config.model.output_dims
            },

            settings: Settings {
                something: config.settings.something
            }
        }
    )
}
