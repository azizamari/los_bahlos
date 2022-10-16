import { Injectable } from '@angular/core';
import { HttpClient } from "@angular/common/http";
@Injectable({
  providedIn: 'root'
})
export class ComentsService {

  constructor(private http:HttpClient) { }
  getcoments(){
    return this.http.get("http://localhost:8000/quizzes")
  }
}

