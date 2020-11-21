use tch;
use tch::{CModule, Tensor};
// use tch::vision::imagenet;

type Error = Box<dyn std::error::Error>;

#[derive(Debug)]
pub struct TorchSession {
    pub environment: CModule,
}


impl TorchSession  {
    /// Load torchscript from file and prepare torch runtime
    ///
    pub fn new(model_file: &str) -> Self {
        let environment = tch::CModule::load(model_file).unwrap();

        TorchSession {
            environment
        }
    }


    pub fn run(self, sample: Vec<f32>) -> Result<Vec<f32>, Error> {
        let input_data = Tensor::of_slice(&sample[..]);
        let output = input_data
        .unsqueeze(0)
        .apply(&self.environment)
        .softmax(-1, tch::Kind::Float);

        Ok(vec![])
    }
}



#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {

        assert_eq!(2 + 2, 4);
    }
}