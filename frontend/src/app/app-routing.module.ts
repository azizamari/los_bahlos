import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomepageComponent } from './components/homepage/homepage.component'; 
import { ContactComponent } from './components/contact/contact.component';
import { AboutComponent } from './components/about/about.component';
import { QuizComponent } from './components/quiz/quiz.component';
const routes: Routes = [
  {path : 'home' , component:HomepageComponent },
  {path : 'contact' , component:ContactComponent },
  {path : 'about' , component:AboutComponent },
  {path : 'quiz' , component:QuizComponent },
];
@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
