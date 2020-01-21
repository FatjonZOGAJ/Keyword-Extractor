import Vue from 'vue';
import Router from 'vue-router';
import Demo from './components/Demo.vue';
import Ping from './components/Ping.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      name: 'Demo',
      component: Demo,
    },
    {
      path: '/ping',
      name: 'Ping',
      component: Ping,
    },
  ],
});
