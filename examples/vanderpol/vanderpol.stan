functions {
  vector my_vanderpol(vector initial_conditions);
}

data {
  vector[2] observed;
}

parameters {
  vector[2] initial_conditions;
  real<lower = 0, upper = 0.2> sigma[2];
}

transformed parameters {
  vector[2] simulated = my_vanderpol(initial_conditions);
}

model {
  target += normal_lpdf(initial_conditions | simulated, sigma);
}

