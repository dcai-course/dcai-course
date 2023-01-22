"use strict";

(() => {
  const params = new URLSearchParams(window.location.search);

  if (params.has('lecnotes')) {
    Array.from(document.getElementsByClassName('lecnote')).forEach((el) => {
      el.style = 'display: block;'
    })
  }
})()
