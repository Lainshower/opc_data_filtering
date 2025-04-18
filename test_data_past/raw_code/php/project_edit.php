<?php

$id = isset($vars[1]) ? $vars[1] : null;
$object = Project::findById($id);
if (is_null($object)) {
  HTML::forward('core/404');
}

// bootstrap field widgets
FormWidgetImage::bootstrap('images');
FormWidgetImage::bootstrap('thumbnail');
FormWidgetDatepicker::bootstrap('date');
FormWidgetPlupfile::bootstrap('attachment');
FormWidgetPlupfile::bootstrap('application');

// handle form submission
if (isset($_POST['submit'])) {
  $error_flag = false;

  /// validation
  
  // validation for $title
  $title = isset($_POST["title"]) ? strip_tags($_POST["title"]) : null;
  if (empty($title)) {
    Message::register(new Message(Message::DANGER, i18n(array("en" => "title is required.", "zh" => "请填写title"))));
    $error_flag = true;
  }

  if (strlen($title) >= 150) {
    Message::register(new Message(Message::DANGER, i18n(array("en" => "Max length for title is 150", "zh" => "title 不能超过150个字符"))));
    $error_flag = true;
  }
  
  // validation for $password
  $password = isset($_POST["password"]) ? strip_tags($_POST["password"]) : null;
  $retype_password = isset($_POST["retype_password"]) ? strip_tags($_POST["retype_password"]) : null;
  if (empty($password)) {
    Message::register(new Message(Message::DANGER, i18n(array("en" => "password is required.", "zh" => "请填写password"))));
    $error_flag = true;
  }

  if (strlen($password) >= 15) {
    Message::register(new Message(Message::DANGER, i18n(array("en" => "Max length for password is 15", "zh" => "password 不能超过15个字符"))));
    $error_flag = true;
  }

  if ($password != $retype_password) {
    Message::register(new Message(Message::DANGER, i18n(array("en" => "Retype value does not match for password", "zh" => "再次输入的password与原值不匹配"))));
    $error_flag = true;
  }
  
  // validation for $email
  $email = isset($_POST["email"]) ? strip_tags($_POST["email"]) : null;
  $retype_email = isset($_POST["retype_email"]) ? strip_tags($_POST["retype_email"]) : null;
  if (empty($email)) {
    Message::register(new Message(Message::DANGER, i18n(array("en" => "email is required.", "zh" => "请填写email"))));
    $error_flag = true;
  }

  if (strlen($email) >= 30) {
    Message::register(new Message(Message::DANGER, i18n(array("en" => "Max length for email is 30", "zh" => "email 不能超过30个字符"))));
    $error_flag = true;
  }

  if ($email != $retype_email) {
    Message::register(new Message(Message::DANGER, i18n(array("en" => "Retype value does not match for email", "zh" => "再次输入的email与原值不匹配"))));
    $error_flag = true;
  }
  
  // validation for $description_en
  $description_en = isset($_POST["description_en"]) ? $_POST["description_en"] : null;  
  // validation for $description_zh
  $description_zh = isset($_POST["description_zh"]) ? $_POST["description_zh"] : null;
  if (empty($description_zh)) {
    Message::register(new Message(Message::DANGER, i18n(array("en" => "description_zh is required.", "zh" => "请填写description_zh"))));
    $error_flag = true;
  }
  
  // validation for $active
  $active = isset($_POST["active"]) ? 1 : 0;  
  // validation for $price
  $price = isset($_POST["price"]) ? strip_tags($_POST["price"]) : null;  
  // validation for $images
  $images = isset($_POST["images"]) ? strip_tags(trim($_POST["images"])) : null;  
  // validation for $thumbnail
  $thumbnail = isset($_POST["thumbnail"]) ? strip_tags(trim($_POST["thumbnail"])) : null;
  if (empty($thumbnail)) {
    Message::register(new Message(Message::DANGER, i18n(array("en" => "thumbnail is required.", "zh" => "请填写thumbnail"))));
    $error_flag = true;
  }
  
  // validation for $date
  $date = isset($_POST["date"]) ? strip_tags($_POST["date"]) : null;
  if (empty($date)) {
    Message::register(new Message(Message::DANGER, i18n(array("en" => "date is required.", "zh" => "请填写date"))));
    $error_flag = true;
  }
  
  // validation for $attachment
  $attachment = isset($_POST["attachment"]) ? strip_tags(trim($_POST["attachment"])) : null;
  // check not empty
  if (empty($attachment)) {
    Message::register(new Message(Message::DANGER, i18n(array("en" => "attachment is required.", "zh" => "请填写attachment"))));
    $error_flag = true;
  }

  // check upload_dir
  if (!is_dir(WEBROOT . DS . "files/plupfiletest")) {
    mkdir(WEBROOT . DS . "files/plupfiletest");
  }
  if (!is_writable(WEBROOT . DS . "files/plupfiletest")) {
    $error_flag = true;
    Message::register(new Message(Message::DANGER, i18n(array("en" => "Upload dir is not writable.", "zh" => "上传文件夹不可写"))));
  } else {
    $files = explode("\n", trim($attachment));
    // check max_file_number
    if (sizeof($files) > 3) {
      Message::register(new Message(Message::DANGER, i18n(array("en" => "Max file allowed to be uploed is 3. Please reduce uploaed files.", "zh" => "您最多可以上传3个文件，请减少上传的文件数量"))));
      $error_flag = true;
    }
    // check file extension
    foreach ($files as $file) {
      $file = trim($file);
      $tokens = explode(".", $file);
      $extension = array_pop($tokens);
      if (!in_array(strtolower($extension), array('jpg','png','gif','zip'))) {
        Message::register(new Message(Message::DANGER, i18n(array("en" => "Only file with extension jpg,png,gif,zip is allowed. Please restrict your files with these types.", "zh" => "上传文件仅支持jpg,png,gif,zip，请检查您的上传文件"))));
        $error_flag = true;
        break;
      }
    }
  }
  
  // validation for $application
  $application = isset($_POST["application"]) ? strip_tags(trim($_POST["application"])) : null;
  // check upload_dir
  if (!is_dir(WEBROOT . DS . "files/plupfiletest")) {
    mkdir(WEBROOT . DS . "files/plupfiletest");
  }
  if (!is_writable(WEBROOT . DS . "files/plupfiletest")) {
    $error_flag = true;
    Message::register(new Message(Message::DANGER, i18n(array("en" => "Upload dir is not writable.", "zh" => "上传文件夹不可写"))));
  } else {
    $files = explode("\n", trim($application));
    // check max_file_number
    if (sizeof($files) > 1) {
      Message::register(new Message(Message::DANGER, i18n(array("en" => "Max file allowed to be uploed is 1. Please reduce uploaed files.", "zh" => "您最多可以上传1个文件，请减少上传的文件数量"))));
      $error_flag = true;
    }
    // check file extension
    foreach ($files as $file) {
      $file = trim($file);
      $tokens = explode(".", $file);
      $extension = array_pop($tokens);
      if (!in_array(strtolower($extension), array('jpg','png','gif','zip'))) {
        Message::register(new Message(Message::DANGER, i18n(array("en" => "Only file with extension jpg,png,gif,zip is allowed. Please restrict your files with these types.", "zh" => "上传文件仅支持jpg,png,gif,zip，请检查您的上传文件"))));
        $error_flag = true;
        break;
      }
    }
  }
  /// proceed submission
  
  // proceed for $title
  $object->setTitle($title);
  
  // proceed for $password
  $object->setPassword($password);
  
  // proceed for $email
  $object->setEmail($email);
  
  // proceed for $description_en
  $object->setDescriptionEn($description_en);
  
  // proceed for $description_zh
  $object->setDescriptionZh($description_zh);
  
  // proceed for $active
  $object->setActive($active);
  
  // proceed for $price
  if (!empty($price)) {
    $object->setPrice($price);
  }
  
  // proceed for $images
  $object->setImages($images);
  
  // proceed for $thumbnail
  $object->setThumbnail($thumbnail);
  
  // proceed for $date
  $object->setDate($date/1000);
  
  // proceed for $attachment
  $files = explode("\n", trim($attachment));
  $rtn = array();
  foreach ($files as $file) {
    $file = trim($file);
    // for cache file, we move it to its proper location 
    if (strpos($file, str_replace(WEBROOT . DS, "", CACHE_DIR)) === 0) {
      $oldname = WEBROOT . DS . $file;
      $newname = WEBROOT . DS . "files/plupfiletest" . str_replace(CACHE_DIR, "", WEBROOT . DS . $file);
      rename($oldname, $newname);
      $file = str_replace(WEBROOT . DS, "", $newname);
    }
    $rtn[] = $file;
  }
  $object->setAttachment(implode("\n", $rtn));
  
  // proceed for $application
  $files = explode("\n", trim($application));
  $rtn = array();
  foreach ($files as $file) {
    $file = trim($file);
    // for cache file, we move it to its proper location 
    if (strpos($file, str_replace(WEBROOT . DS, "", CACHE_DIR)) === 0) {
      $oldname = WEBROOT . DS . $file;
      $newname = WEBROOT . DS . "files/plupfiletest" . str_replace(CACHE_DIR, "", WEBROOT . DS . $file);
      rename($oldname, $newname);
      $file = str_replace(WEBROOT . DS, "", $newname);
    }
    $rtn[] = $file;
  }
  $object->setApplication(implode("\n", $rtn));
  if ($error_flag == false) {
    if ($object->save()) {
      Message::register(new Message(Message::SUCCESS, i18n(array("en" => "Record saved", "zh" => "记录保存成功"))));
      HTML::forwardBackToReferer();
    } else {
      Message::register(new Message(Message::DANGER, i18n(array("en" => "Record failed to save", "zh" => "记录保存失败"))));
    }
  }
}



$html = new HTML();

$html->renderOut('core/backend/html_header', array(
  'title' => i18n(array(
  'en' => 'Edit Project',
  'zh' => 'Edit 项目',
  )),
));
$html->output('<div id="wrapper">');
$html->renderOut('core/backend/header');


$html->renderOut('project/backend/project_edit', array(
  'object' => $object
));


$html->output('</div>');

$html->renderOut('core/backend/html_footer');

exit;

