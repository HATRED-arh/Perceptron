fn main() {
    let dataset = vec![
        [2.7810836, 2.550537003, 0.0],
        [1.465489372, 2.362125076, 0.0],
        [3.396561688, 4.400293529, 0.0],
        [1.38807019, 1.850220317, 0.0],
        [3.06407232, 3.005305973, 0.0],
        [7.627531214, 2.759262235, 1.0],
        [5.332441248, 2.088626775, 1.0],
        [6.922596716, 1.77106367, 1.0],
        [8.675418651, -0.242068655, 1.0],
        [7.673756466, 3.508563011, 1.0],
    ];
    let weights = train_weights(&dataset, 0.1, 5);
    println!("{:?}", weights);
}
fn predict(row: &[f64; 3], weights: &[f64]) -> f64 {
    let mut activation = weights[0];
    for i in 0..row.len() - 1 {
        activation += weights[i + 1] * row[i]
    }
    if activation >= 0.0 {
        1.0
    } else {
        0.0
    }
}

fn train_weights(dataset: &[[f64; 3]], learning_rate: f64, num_epoch: u32) -> Vec<f64> {
    let mut weights = vec![0.0; dataset[0].len()];
    for epoch in 0..num_epoch {
        let mut sum_error = 0.0;
        for row in dataset {
            let prediction = predict(row, &weights);
            let error = row.last().unwrap() - prediction;
            sum_error += error.powi(2);
            weights[0] += learning_rate * error;
            for i in 0..weights.len() - 1 {
                weights[i + 1] += learning_rate * error * row[i];
            }
        }
        println!(
            "Epoch {}, lrate {}, error {}",
            epoch, learning_rate, sum_error
        )
    }
    weights
}
