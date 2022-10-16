import { Component,OnInit } from '@angular/core';
import { Comments } from './classes/comments';
import { ComentsService } from './services/coments.service';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'GoMyCode2022';
  lstcomments: any=[];
  constructor(private freeapiservice : ComentsService){}
  getcoments(){
    this.freeapiservice.getcoments().subscribe((data:any)=>{
           this.lstcomments=data
           console.log(this.lstcomments)
    })
  }
  ngOnInit():void{
   this.getcoments()
   console.log('hheee')
  }
}
