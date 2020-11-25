use tch;
use tch::{CModule, Tensor};
// use tch::vision::imagenet;
use std::sync::Arc;

type Error = Box<dyn std::error::Error>;

#[derive(Debug, Clone)]
pub struct TorchSession {
    pub environment: Arc<CModule>,
}

// impl Copy for TorchSession {}
// impl Clone for TorchSession {
//     fn clone(&self) -> TorchSession {
//         *self
//     }
// }


impl TorchSession  {
    /// Load torchscript from file and prepare torch runtime
    pub fn new(model_file: &str) -> Self {
        let environment = Arc::new(tch::CModule::load(model_file).unwrap());
        // let environment = tch::CModule::load(model_file).unwrap();

        TorchSession {
            environment
        }
    }

    pub fn run(&self, sample: Vec<f32>) -> Result<Vec<f32>, Error> {
        let input_data = Tensor::of_slice(&sample[..]);
        let environment = self.environment.clone();

        // let output1 = environment
        //     .forward_ts(&[input_data.shallow_clone()])
        //     .unwrap()
        //     .softmax(-1, tch::Kind::Float);
        // dbg!("from TorchSession preds1: {}", &output1);

        let output = input_data
            .unsqueeze(0)
            .apply(&*environment)
            .softmax(-1, tch::Kind::Float);

        dbg!("from TorchSession preds: {}", &output);

        // let output = input_data
        // .unsqueeze(0)
        // .apply(&self.environment)
        // .softmax(-1, tch::Kind::Float);

        // TODO return output
        // let some: Vec<_> = output.values().iter();
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
