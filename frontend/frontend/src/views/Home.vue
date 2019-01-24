<template lang="pug">
.container
  a-row.row(type="flex" justify="center" align="middle")
    a-col(:span="14" :offset="4")
      a-textarea.text(placeholder="Please input a sentence" :autosize="{ minRows: 3, maxRows: 6 }" v-model="inputText")
    a-col(:span="4")
      a-button(type="primary" icon="forward" size="large" @click="classify") Check your MBTI
  a-row.row(type="flex" justify="center" align="middle")
    a-col(:span="14")
      .label-large Your MBTI is: &nbsp;&nbsp;&nbsp;&nbsp;{{ label0 }}
  a-row.row(type="flex" justify="center" align="middle")
    a-col(:span="14" :offset="4")
      .label Please input the MBTI you want to transform into:
      a-row(type="flex" justify="space-around" align="middle")
        a-col.radio-container(:span="5")
          a-radio-group(@change="v => onChange(v, 0)" v-model="chosen[0]")
            a-radio(:style="radioStyle" value="I") Introversion
            a-radio(:style="radioStyle" value="E") Extraversion
        a-col.radio-container(:span="5")
          a-radio-group(@change="v => onChange(v, 1)" v-model="chosen[1]")
            a-radio(:style="radioStyle" value="N") Intuition
            a-radio(:style="radioStyle" value="S") Sensing
        a-col.radio-container(:span="5")
          a-radio-group(@change="v => onChange(v, 2)" v-model="chosen[2]")
            a-radio(:style="radioStyle" value="T") Thinking
            a-radio(:style="radioStyle" value="F") Feeling
        a-col.radio-container(:span="5")
          a-radio-group(@change="v => onChange(v, 3)" v-model="chosen[3]")
            a-radio(:style="radioStyle" value="J") Judgment
            a-radio(:style="radioStyle" value="P") Perception
    a-col(:span="4")
      a-button(type="primary" icon="edit" size="large" @click="run") Change language style
  a-row.row(type="flex" justify="center" align="middle")
    a-col(:span="14")
      a-textarea.text(placeholder="Output of transformation..." :autosize="{ minRows: 3, maxRows: 6 }" v-model="result" :disabled="true")

</template>


<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { message } from 'ant-design-vue';
import api from '@/api';

@Component
export default class Home extends Vue {
  inputText: string = ''

  label0: string = '____'

  result: string = ''

  chosen: string[] = ['I', 'N', 'T', 'J']

  radioStyle = {
    display: 'block',
    height: '50px',
    lineHeight: '50px',
    fontSize: '18px',
  }

  classify() {
    api.getClass({ text: this.inputText })
      .then(({ data }) => {
        if (data.info) {
          this.label0 = data.label;
        } else {
          message.error('Network Error');
        }
      }).catch((e: any) => {
        message.error('Network Error');
      });
  }

  onChange(v: any, id: number) {
    this.chosen[id] = v.target.value;
  }

  run() {
    const label1: string = `${this.chosen[0]}${this.chosen[1]}${this.chosen[2]}${this.chosen[3]}`;
    console.log(label1);
    api.getToken({ text: this.inputText, label0: this.label0, label1 })
      .then(({ data }) => {
        if (data.info) {
          this.result = data.res;
        } else {
          message.error('Network Error');
        }
      }).catch((e: any) => {
        message.error('Network Error');
      });
  }
}
</script>


<style lang="stylus">
.container
  padding 100px
  display flex;
  flex-direction column;
  align-content center;
  justify-items center;
  .row
    width 100%
    margin 30px 0
  .text
    heigth 200px
    font-size 24px
    line-height 36px
    box-shadow 0 -1px 0 #e0e0e0,0 0 1px rgba(0,0,0,.12),0 2px 4px rgba(0,0,0,.24)
    border 2px solid white
  .label-large
    font-size 30px
    line-height 45px
  .label
    font-size 24px
    line-height 36px
  .radio-container
    box-shadow 0 -1px 0 #e0e0e0,0 0 1px rgba(0,0,0,.12),0 2px 4px rgba(0,0,0,.24)
    border 2px solid white
    padding 30px 15px
    margin 25px 0
</style>
