import Vue from 'vue';
import {
  Steps, Row, Col, Tabs, Form, Input, Breadcrumb,
  Icon, Select, Slider, InputNumber, Spin, Modal,
  Avatar, Dropdown, Menu, Carousel, Timeline, Button,
  Radio,
} from 'ant-design-vue';

import App from './App.vue';
import router from './router';

Vue.use(Steps);
Vue.use(Row);
Vue.use(Col);
Vue.use(Tabs);
Vue.use(Form);
Vue.use(Input);
Vue.use(Icon);
Vue.use(Select);
Vue.use(Slider);
Vue.use(InputNumber);
Vue.use(Spin);
Vue.use(Breadcrumb);
Vue.use(Modal);
Vue.use(Avatar);
Vue.use(Dropdown);
Vue.use(Menu);
Vue.use(Carousel);
Vue.use(Timeline);
Vue.use(Button);
Vue.use(Radio);


Vue.config.productionTip = false;

new Vue({
  router,
  render: h => h(App),
}).$mount('#app');
