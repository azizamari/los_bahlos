import { Injectable } from '@angular/core';
import { HttpClient } from "@angular/common/http";
@Injectable({
  providedIn: 'root'
})
export class ComentsService {

  constructor(private http:HttpClient) { }
  getcoments(){
    return this.http.get("https://jsonplaceholder.typicode.com/posts/1/comments")
  }
}

