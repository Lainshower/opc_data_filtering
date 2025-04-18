package model

type ProjectList struct {
	Address		string 		`json:"href"`
	ItemsList	[]Item 		`json:"items"`
}

type Item struct {
	Id 			string 		`json:"id"`
	Name 		string		`json:"name"`
}