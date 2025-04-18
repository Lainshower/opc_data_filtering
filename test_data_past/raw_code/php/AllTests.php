<?php
namespace tests;

use PHPUnit_Framework_TestSuite;
use tests\app\ApplicationTests;

class AllTests
{
    public static function suite()
    {
        $suite = new PHPUnit_Framework_TestSuite('All Tests');

        $suite->addTestSuite(ApplicationTests::suite());

        return $suite;
    }
}
 