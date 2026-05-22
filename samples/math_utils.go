package main

import (
	"fmt"
	"math"
)

// Point represents a 2D coordinate.
type Point struct {
	X, Y float64
}

// Distance calculates the Euclidean distance between two points.
// It uses the Pythagorean theorem: sqrt((x2-x1)^2 + (y2-y1)^2).
func (p Point) Distance(other Point) float64 {
	dx := p.X - other.X
	dy := p.Y - other.Y
	return math.Sqrt(dx*dx + dy*dy)
}

// AreaOfCircle calculates the area given a radius.
func AreaOfCircle(radius float64) float64 {
	return math.Pi * math.Pow(radius, 2)
}

func main() {
	p1 := Point{0, 0}
	p2 := Point{3, 4}
	fmt.Printf("Distance: %.2f\n", p1.Distance(p2))
}
