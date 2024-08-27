estimate <- function(n) { return (4 * sum((runif(n)^2 + runif(n)^2) < 1) / n) }
n<-10000000
estimate(n)

estimateEllipseArea <- function(a, b, n) {

  # Generate random points within a bounding rectangle
  x_points <- runif(n, -a, a)
  y_points <- runif(n, -b, b)
  
  # Check if each point is inside the ellipse
  points_inside_ellipse <- (x_points^2 / a^2 + y_points^2 / b^2) < 1
  
  # Calculate the ratio of points inside the ellipse to the total number of points
  area_ratio <- sum(points_inside_ellipse) / n
  
  # Estimate the area of the ellipse
  ellipse_area <- 4 * a * b * area_ratio
  
  return(ellipse_area)
}

a <- 5  # Semi-major axis
b <- 3  # Semi-minor axis
n <- 10000000  # Number of random points

estimated_area <- estimateEllipseArea(a, b, n)
print(estimated_area)
