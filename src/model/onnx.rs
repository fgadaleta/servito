use ndarray::Array;
use onnxruntime::{environment::Environment,
                  tensor::OrtOwnedTensor,
                  GraphOptimizationLevel,
                  LoggingLevel};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;


type Error = Box<dyn std::error::Error>;

#[derive(Debug)]
pub struct OnnxSession {
    pub model_filepath: String,
    pub environment: Environment,
    // pub input_shape: Vec<usize>,
    // pub output_shape: Vec<usize>
}

impl OnnxSession {
    pub fn new(model_filepath: String) -> Result<OnnxSession, Error> {
        let subscriber = FmtSubscriber::builder()
            .with_max_level(Level::TRACE)
            .finish();

        let environment = Environment::builder()
            .with_name("test_environment")
            // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
            .with_log_level(LoggingLevel::Info)
            .build()?;

        tracing::subscriber::set_global_default(subscriber)
            .expect("setting default subscriber failed");

        Ok(OnnxSession {
            model_filepath: model_filepath.to_owned(),
            environment,
            // input_shape,
            // output_shape
        })
    }

    pub fn run(&self, sample: Vec<f32>) -> Result<Vec<f32>, Error> {
        // tracing::subscriber::set_global_default(self.subscriber).expect("setting default subscriber failed");

        let mut session = self
            .environment
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::Basic)?
            .with_number_threads(1)?
            .with_model_from_file(self.model_filepath.clone())?;

        let mut input_shape: Vec<usize> = session.inputs[0]
            .dimensions()
            .map(|d| {
                let curdim = match d {
                    Some(dim) => dim,
                    None => 1,
                };
                curdim
            })
            .collect();

        println!("input_0: {:?}", input_shape);

        let output_shape: Vec<usize> = session.outputs[0]
            .dimensions()
            .map(|d| {
                let curdim = match d {
                    Some(dim) => dim,
                    None => 1,
                };
                curdim
                // d.unwrap()
            })
            .collect();
        println!("output_0: {:?}", output_shape);

        // TODO if dimensions present in toml config, assert it matches
        // assert_eq!(input0_shape, [1, 3, 224, 224]);
        // assert_eq!(output0_shape, [1, 1000, 1, 1]);

        // total input dimensions
        let mut n = 1;
        for el in input_shape.iter_mut() {
            n *= *el;
        }
        dbg!("total input dims: {}", n);

        // Prepare sample and infer into runtime session
        let sample = Array::from(sample).into_shape(input_shape).unwrap();
        let sample = vec![sample];
        let predictions: Vec<OrtOwnedTensor<f32, _>> = session.run(sample).unwrap();

        // Format predictions and return
        let mut preds: Vec<f32> = vec![];
        for tensor in predictions {
            for pred in tensor.iter() {
                preds.push(*pred);
            }
        }
        Ok(preds)
    }
}
