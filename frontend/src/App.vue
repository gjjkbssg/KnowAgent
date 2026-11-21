<script setup lang="ts">
import { useAuthStore } from './stores/auth'

const auth = useAuthStore()
</script>

<template>
  <router-view />
  <t-dialog
    :visible="!!auth.tempPassword"
    header="管理员账号已创建"
    confirm-btn="我已记下"
    :cancel-btn="null"
    :close-btn="false"
    :close-on-overlay-click="false"
    :close-on-esc-keydown="false"
    @confirm="auth.tempPassword = ''"
  >
    <p>邮箱：{{ auth.user?.email }}</p>
    <p>临时密码：<code>{{ auth.tempPassword }}</code></p>
    <p>这个密码只显示这一次。请记下它，并在「设置」里修改密码。</p>
  </t-dialog>
</template>
