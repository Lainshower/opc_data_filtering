import { Injectable } from '@angular/core';
import { HttpService } from './http.service';
import { Http } from '@angular/http';


@Injectable()
export class SeatingService extends HttpService{

  constructor(public http: Http)  {
    super(http);
  }

  view(data) {
    return this.get('layout/encoding', data);
  }

  addLayout(data) {
    return this.post('layout/add', data);
  }

  editLayout(data) {
    return this.post('layout/edit', data);
  }

  deleteLayout(data) {
    return this.post('layout/delete', data);
  }
}
