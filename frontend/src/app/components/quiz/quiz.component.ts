import { Component, Input, OnInit } from '@angular/core';
import { ComentsService } from 'src/app/services/coments.service';

@Component({
  selector: 'app-quiz',
  templateUrl: './quiz.component.html',
  styleUrls: ['./quiz.component.css']
})
export class QuizComponent implements OnInit {
  lstcomments: any=[];
  question: any=[];
  
  constructor(private freeapiservice : ComentsService){}

  getcoments(){
    this.freeapiservice.getcoments().subscribe((data:any)=>{
           this.lstcomments=data
           console.log(data);
           
           this.lstcomments.map((element:any)=>{
            this.question.push(element.Questions)
            console.log(this.question);
            
           })
    })
  }
  
  ngOnInit():void{

   this.getcoments()
  }

}
