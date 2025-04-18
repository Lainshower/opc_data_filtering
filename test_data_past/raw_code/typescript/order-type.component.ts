import { Component, Input, OnInit } from '@angular/core';
import { OrderType } from '../order.model';

@Component({
  selector: 'app-order-type',
  templateUrl: './order-type.component.html',
  styleUrls: ['./order-type.component.css']
})
export class OrderTypeComponent implements OnInit {
  @Input('type') type: OrderType;
  types = OrderType;

  constructor() { }

  ngOnInit() {
  }

}
