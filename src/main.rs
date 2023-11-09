/*
* Basic idea of a neural network
* Have a weight w and bias b(both found via training by minimising the cost function)
* Start with random weight
* Primitive cost function: add up the (target_value - predicted_value)**2
* change it by an epsilon and see how the cost function changes(derivative of cost function)
*/

use rand::Rng;
// Eventually replace rand with own random number generator

#[warn(dead_code)]
struct TrainingElem {
    input_values: Vec<f32>,
    output_value: f32,
}

impl TrainingElem {
    pub fn new(inputs: Vec<f32>, expected_out: f32) -> Self {
        return TrainingElem {
            input_values: inputs,
            output_value: expected_out,
        };
    }
}

fn cost(training_data_vec: &Vec<TrainingElem>, w1: f32, w2: f32) -> f32 {
    let mut error = 0.0;
    for training_data in training_data_vec.iter() {
        let current_prediction =
            w1 * (training_data.input_values[0]) + w2 * (training_data.input_values[1]);
        error += (training_data.output_value - current_prediction)
            * (training_data.output_value - current_prediction);
    }
    error / (training_data_vec.len() as f32)
}

fn train_n(training_data: &Vec<TrainingElem>, mut w1: f32, mut w2: f32) -> (f32, f32) {
    let learning_rate = 1e-3;
    let training_iter = 1000000;
    for _ in 0..training_iter {
        let cost_fit = cost(&training_data, w1, w2);
        println!("{:?}", cost_fit);
        let partial_deriv_w1 = (cost(&training_data, w1 + learning_rate, w2)
            - cost(&training_data, w1, w2))
            / (learning_rate);
        let partial_deriv_w2 = (cost(&training_data, w1, w2 + learning_rate)
            - cost(&training_data, w1, w2))
            / (learning_rate);
        // let deriv_mod = partial_deriv_w1 * partial_deriv_w1 + partial_deriv_w2 * partial_deriv_w2;
        w1 -= partial_deriv_w1 * learning_rate;
        w2 -= partial_deriv_w2 * learning_rate;
    }
    return (w1, w2);
}

fn sigmoid(input: f32) -> f32 {
    1.0 / (1.0 + (-input).exp())
}
fn relu(input: f32) -> f32 {
    match input < 0.0 {
        true => 0.0,
        false => input,
    }
}

fn main() {
    let and_training_data: Vec<TrainingElem> = vec![
        TrainingElem::new(vec![0.0, 0.0], 0.0),
        TrainingElem::new(vec![0.0, 1.0], 0.0),
        TrainingElem::new(vec![1.0, 0.0], 0.0),
        TrainingElem::new(vec![1.0, 1.0], 1.0),
    ];
    let mut rng = rand::thread_rng();
    let w1: f32 = rng.gen_range(0.0..10.0);
    let w2: f32 = rng.gen_range(0.0..10.0);
    let (weight_1, weight_2) = train_n(&and_training_data, w1, w2);
    println!("Predicted or values");
    for training_data in and_training_data.iter() {
        println!(
            "And value for {:?} and {:?}, post relu is {:?}, post sigmoid: {:?}",
            training_data.input_values[0],
            training_data.input_values[1],
            relu(
                weight_1 * training_data.input_values[0] + weight_2 * training_data.input_values[1]
            ),
            sigmoid(
                weight_1 * training_data.input_values[0] + weight_2 * training_data.input_values[1]
            ),
        );
    }
}
