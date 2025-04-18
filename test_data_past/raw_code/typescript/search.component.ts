import { Component, Inject, OnInit, Renderer2 } from '@angular/core';
import { FormControl } from '@angular/forms';
import { animationsList } from "../searchList.js";

@Component({
  selector: 'app-search',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent implements OnInit {

  form = new FormControl();

  public searchedList = [];
  public isDisplayAllList = false
  
  constructor(private renderer: Renderer2) {}

  ngOnInit(): void {}
  
  // タイトルから検索
  public onSearchByTitle(keyword: string) {
    let parent = this.renderer.parentNode('resultZone');
    this.renderer.removeChild(parent, 'a');
    // 表示リストを初期化
    this.searchedList = [];
    // 検索結果表示
    animationsList.forEach(animation => {
      if (animation.title.indexOf(keyword) != -1) {
        let searchedAnimation = `${animation.title}：${animation.season}`;
        this.searchedList.push(searchedAnimation);
      }
    });
  }
  
  // 放送時期から検索
  public onSearchBySeason(keyword: string) {
    // 表示リストを初期化
    this.searchedList = [];
    // 検索結果表示
    animationsList.forEach(animation => {
      if (animation.season.indexOf(keyword) != -1) {
        let searchedAnimation = `${animation.title}：${animation.season}`;
        this.searchedList.push(searchedAnimation);
      }
    });
    return;
  }

  // 一覧表示を制御
  public onDisplayAllList() {
    this.isDisplayAllList = true;
  }
  public onHiddenAllList() {
    this.isDisplayAllList = false;
  }
  
  // 表示をリセット
  public onReset() {
    this.searchedList = [];
  }

}