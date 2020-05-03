functions {
  vector my_rosenbrock(vector xy);
}

parameters {
  vector[2] xy;
}

model {
  target += -my_rosenbrock(xy);
}

